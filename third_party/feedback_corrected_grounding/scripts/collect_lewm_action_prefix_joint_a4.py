#!/usr/bin/env python3
"""Replay accepted A4 bundles into raw seven-task RGB/geometry memmaps."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time
from typing import Any, Mapping, Sequence

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_a4_storage import load_a4_state_cache  # noqa: E402
from fcpg.lewm_action_branch_a4 import BRANCH_COUNT, HORIZON  # noqa: E402
from fcpg.lewm_action_branch_a4_collection import (  # noqa: E402
    _bounded_expert_action,
    a4_camera_name,
    make_a4_episode_runtime,
)
from fcpg.lewm_action_prefix_joint_data import (  # noqa: E402
    SCHEMA_VERSION,
    phase_labels_for_bundle_indices,
    replay_index_payload,
    select_balanced_bundle_indices,
    select_phase_balanced_bundle_indices,
)
from fcpg.lewm_orbit_camera import (  # noqa: E402
    ORBIT_DEVELOPMENT_VIEWS,
    ORBIT_TRAIN_VIEWS,
    OrbitCameraView,
    orbit_view_from_values,
)
from fcpg.lewm_counterfactual_runtime import (  # noqa: E402
    capture_branch_snapshot,
    corrected_points,
    grounded_valid_mask,
    restore_branch_snapshot,
)
from fcpg.transformer_grounding import load_fcpg_vit_adapter  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_active_control_a4.json"
DEFAULT_CACHE = ROOT / "outputs/lewm_action_branch_a4/formal_v2/state_cache.npz"
DEFAULT_LEDGER = ROOT / "outputs/lewm_action_branch_a4/formal_v2/proposal_ledger.npz"
DEFAULT_CORRECTOR = ROOT / (
    "outputs/transformer_residual/lewm_stage1/shared_4task_seed0_sig0/"
    "fcpg_transformer_residual_best.pt"
)
DEFAULT_OUTPUT = ROOT / "outputs/lewm_action_prefix_a4/joint_raw_24train_8dev"
RAW_STEPS_PER_MODEL_STEP = 5
MULTIVIEW_CAMERA_NAMES = ("corner", "corner2", "corner3", "corner4", "topview")
GROUNDING_MODES = ("fcpg", "nominal")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--corrector", type=Path, default=DEFAULT_CORRECTOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-per-task", type=int, default=24)
    parser.add_argument("--development-per-task", type=int, default=8)
    parser.add_argument(
        "--phase-schedule",
        type=Path,
        default=None,
        help=(
            "Use an A4c phase schedule to spread selected bundles over each "
            "episode rather than taking its earliest proposal."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--camera-view",
        action="append",
        choices=MULTIVIEW_CAMERA_NAMES,
        help="Repeat each selected physical bundle for this fixed world camera.",
    )
    parser.add_argument(
        "--all-fixed-views",
        action="store_true",
        help="Use corner, corner2, corner3, corner4, and topview for every task.",
    )
    parser.add_argument(
        "--orbit-clock-panel",
        action="store_true",
        help="Train on even clock hours and develop on held-out odd clock hours.",
    )
    parser.add_argument(
        "--grounding-mode",
        choices=GROUNDING_MODES,
        default="fcpg",
        help="Use the frozen FCPG corrector or exact nominal anchors for the pilot.",
    )
    parser.add_argument(
        "--augment-gripper-actions",
        action="store_true",
        help="Add deterministic open/close/hold/toggle schedules across the eight branches.",
    )
    args = parser.parse_args()
    if args.train_per_task <= 0 or args.development_per_task <= 0:
        parser.error("per-task counts must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    camera_modes = int(bool(args.all_fixed_views)) + int(bool(args.camera_view)) + int(
        bool(args.orbit_clock_panel)
    )
    if camera_modes > 1:
        parser.error("fixed, explicit, and orbit camera modes are mutually exclusive")
    if args.orbit_clock_panel and args.grounding_mode != "nominal":
        parser.error("--orbit-clock-panel currently requires --grounding-mode nominal")
    return args


def _camera_views(args: argparse.Namespace) -> tuple[str, ...]:
    if args.all_fixed_views:
        return MULTIVIEW_CAMERA_NAMES
    if args.camera_view:
        values = tuple(str(value) for value in args.camera_view)
        if len(set(values)) != len(values):
            raise ValueError("camera views must be unique")
        return values
    return ()


def augment_gripper_action_branches(future_action: np.ndarray) -> np.ndarray:
    """Add counterfactual gripper schedules without changing XYZ branch motion."""

    values = np.asarray(future_action, dtype=np.float32)
    if values.shape[-3:] != (BRANCH_COUNT, HORIZON, 20):
        raise ValueError("future action must end in [8,8,20]")
    result = np.array(values, dtype=np.float32, copy=True)
    gripper = np.arange(3, 20, 4, dtype=np.int64)
    result[..., 2, :, gripper] = -1.0
    result[..., 3, :, gripper] = 1.0
    result[..., 4, :, gripper] = 0.0
    result[..., 5, HORIZON // 2 :, gripper] = -1.0
    result[..., 6, HORIZON // 2 :, gripper] = 1.0
    result[..., 7, : HORIZON // 2, gripper] = 1.0
    result[..., 7, HORIZON // 2 :, gripper] = -1.0
    return np.ascontiguousarray(result, dtype=np.float32)


def expand_replay_index_views(
    index: Mapping[str, np.ndarray],
    camera_views: Sequence[str],
) -> dict[str, np.ndarray]:
    """Repeat physical bundle identities over fixed cameras without changing actions."""

    views = tuple(str(value) for value in camera_views)
    if not views:
        return {name: np.array(value, copy=True) for name, value in index.items()}
    if len(set(views)) != len(views) or not set(views) <= set(MULTIVIEW_CAMERA_NAMES):
        raise ValueError("camera view panel is invalid")
    count = int(np.asarray(index["cache_index"]).shape[0])
    repeated: dict[str, np.ndarray] = {}
    for name, value in index.items():
        array = np.asarray(value)
        repeated[name] = np.repeat(array, len(views), axis=0)
    repeated["physical_bundle_index"] = np.repeat(
        np.arange(count, dtype=np.int64), len(views)
    )
    repeated["camera_index"] = np.tile(
        np.arange(len(views), dtype=np.int64), count
    )
    repeated["camera_name"] = np.tile(np.asarray(views), count)
    return repeated


def expand_replay_index_orbit_views(
    index: Mapping[str, np.ndarray],
    *,
    train_views: Sequence[OrbitCameraView] = ORBIT_TRAIN_VIEWS,
    development_views: Sequence[OrbitCameraView] = ORBIT_DEVELOPMENT_VIEWS,
) -> dict[str, np.ndarray]:
    """Assign even clock hours to train and held-out odd hours to development."""

    train_panel = tuple(train_views)
    development_panel = tuple(development_views)
    if not train_panel or not development_panel:
        raise ValueError("orbit train and development panels must be non-empty")
    labels = [view.label for view in train_panel + development_panel]
    if len(labels) != len(set(labels)):
        raise ValueError("orbit camera labels must be unique")
    source_split = np.asarray(index["split"]).astype(str)
    source_rows: list[int] = []
    selected_views: list[OrbitCameraView] = []
    for source_index, split in enumerate(source_split.tolist()):
        panel = train_panel if split == "train" else development_panel
        if split not in {"train", "development"}:
            raise ValueError(f"orbit expansion does not support split {split!r}")
        for view in panel:
            source_rows.append(int(source_index))
            selected_views.append(view)
    source = np.asarray(source_rows, dtype=np.int64)
    expanded = {
        name: np.ascontiguousarray(np.asarray(value)[source])
        for name, value in index.items()
    }
    label_to_index = {label: camera_index for camera_index, label in enumerate(labels)}
    expanded["physical_bundle_index"] = source.copy()
    expanded["camera_index"] = np.asarray(
        [label_to_index[view.label] for view in selected_views], dtype=np.int64
    )
    expanded["camera_name"] = np.asarray([view.label for view in selected_views])
    expanded["camera_base_name"] = np.asarray(
        [a4_camera_name(str(index["task_name"][row])) for row in source_rows]
    )
    expanded["camera_clock_hour"] = np.asarray(
        [view.clock_hour for view in selected_views], dtype=np.int64
    )
    expanded["camera_elevation_degrees"] = np.asarray(
        [view.elevation_degrees for view in selected_views], dtype=np.float32
    )
    expanded["camera_radius_m"] = np.asarray(
        [view.radius_m for view in selected_views], dtype=np.float32
    )
    return expanded


def _cuda0(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("raw A4 collection permits only CUDA device 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.expanduser().resolve(), allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _rgb_uint8(record: Any) -> np.ndarray:
    value = np.asarray(record.rgb)
    if value.ndim != 3 or value.shape[0] != 3:
        raise ValueError("FrameRecord RGB must have shape [3,H,W]")
    if value.dtype == np.uint8:
        return np.ascontiguousarray(value)
    if not np.isfinite(value).all():
        raise ValueError("FrameRecord RGB contains non-finite values")
    return np.ascontiguousarray(np.clip(value * 255.0, 0, 255).round().astype(np.uint8))


def _allocate(path: Path, shape: tuple[int, ...], dtype: np.dtype[Any]) -> np.memmap:
    if path.exists():
        value = np.load(path, mmap_mode="r+")
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise ValueError(f"existing memmap contract drifted for {path.name}")
        return value
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _collect_bundle(
    *,
    task: str,
    task_instance: int,
    environment_seed: int,
    proposal_raw_step: int,
    future_action: np.ndarray,
    width: int,
    height: int,
    benchmark_seed: int,
    camera_name: str | None = None,
    orbit_camera_view: OrbitCameraView | None = None,
) -> tuple[Any, list[list[Any]]]:
    runtime = make_a4_episode_runtime(
        task,
        task_instance,
        environment_seed,
        width=width,
        height=height,
        benchmark_seed=benchmark_seed,
        camera_name=camera_name,
        orbit_camera_view=orbit_camera_view,
    )
    observation = np.asarray(runtime.observation, dtype=np.float32)
    try:
        for raw_step in range(proposal_raw_step + 1):
            if raw_step == proposal_raw_step:
                current = runtime.record_current(observation)
                snapshot = capture_branch_snapshot(runtime.env, include_process_rng=True)
                break
            action = _bounded_expert_action(runtime, observation)
            observation, _reward, terminated, truncated, info = runtime.env.step(action)
            if not isinstance(info, Mapping):
                raise TypeError("A4 source replay info must be a mapping")
            if terminated or truncated:
                raise RuntimeError("A4 source replay ended before accepted proposal")
            observation = np.asarray(observation, dtype=np.float32)
        else:
            raise RuntimeError("A4 proposal step was not reached")

        chunks = np.asarray(future_action, dtype=np.float32)
        if chunks.shape != (BRANCH_COUNT, HORIZON, 20):
            raise ValueError("future action must have shape [8,8,20]")
        raw_actions = chunks.reshape(
            BRANCH_COUNT,
            HORIZON,
            RAW_STEPS_PER_MODEL_STEP,
            4,
        ).reshape(BRANCH_COUNT, HORIZON * RAW_STEPS_PER_MODEL_STEP, 4)
        branch_records: list[list[Any]] = []
        for branch in raw_actions:
            restore_branch_snapshot(runtime.env, snapshot)
            records = []
            for raw_index, action in enumerate(branch, start=1):
                next_observation, _reward, terminated, truncated, info = runtime.env.step(action)
                if terminated or truncated:
                    raise RuntimeError("A4 branch replay ended before H8")
                if not isinstance(info, Mapping):
                    raise TypeError("A4 branch replay info must be a mapping")
                if raw_index % RAW_STEPS_PER_MODEL_STEP == 0:
                    records.append(
                        runtime.record_current(
                            np.asarray(next_observation, dtype=np.float32)
                        )
                    )
            if len(records) != HORIZON:
                raise RuntimeError("A4 branch replay did not produce H1-H8 records")
            branch_records.append(records)
        return current, branch_records
    finally:
        runtime.env.close()


def _record_arrays(
    current: Any,
    branches: list[list[Any]],
    *,
    corrector: Any | None,
    device: torch.device,
    grounding_mode: str = "fcpg",
) -> dict[str, np.ndarray]:
    flat = [current] + [record for branch in branches for record in branch]
    if grounding_mode == "fcpg":
        if corrector is None:
            raise ValueError("FCPG grounding requires a corrector")
        corrected = corrected_points(
            flat,
            SimpleNamespace(corrector=corrector, device=device),
        ).astype(np.float32, copy=False)
    elif grounding_mode == "nominal":
        corrected = np.stack(
            [np.asarray(record.q_nominal, dtype=np.float32) for record in flat]
        )
    else:
        raise ValueError(f"unknown grounding mode {grounding_mode!r}")
    rgb = np.stack([_rgb_uint8(record) for record in flat])
    proprio = np.stack([np.asarray(record.proprio, dtype=np.float32) for record in flat])
    nominal = np.stack([np.asarray(record.q_nominal, dtype=np.float32) for record in flat])
    xyz = np.stack([np.asarray(record.keypoints_xyz, dtype=np.float32) for record in flat])
    valid_nominal = np.stack([np.asarray(record.valid_nominal, dtype=bool) for record in flat])
    q_true = np.stack([np.asarray(record.q_true, dtype=np.float32) for record in flat])
    valid = grounded_valid_mask(
        valid_nominal,
        corrected,
        width=int(rgb.shape[-1]),
        height=int(rgb.shape[-2]),
    )
    future_shape = (BRANCH_COUNT, HORIZON)
    result = {
        "current_rgb": rgb[0],
        "current_proprio": proprio[0],
        "current_q_nominal": nominal[0],
        "current_q_corrected": corrected[0],
        "current_keypoints_xyz": xyz[0],
        "current_valid": valid[0],
        "current_q_true": q_true[0],
        "future_rgb": rgb[1:].reshape(*future_shape, *rgb.shape[1:]),
        "future_proprio": proprio[1:].reshape(*future_shape, *proprio.shape[1:]),
        "future_q_nominal": nominal[1:].reshape(*future_shape, 3, 2),
        "future_q_corrected": corrected[1:].reshape(*future_shape, 3, 2),
        "future_keypoints_xyz": xyz[1:].reshape(*future_shape, 3, 3),
        "future_valid": valid[1:].reshape(*future_shape, 3),
        "future_q_true": q_true[1:].reshape(*future_shape, 3, 2),
    }
    for name, value in result.items():
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"raw replay field {name} contains non-finite values")
    return result


def main() -> int:
    args = _parser()
    device = _cuda0(args.device)
    torch.cuda.set_device(device)
    config = _json(args.config)
    collection = config.get("collection")
    if not isinstance(collection, Mapping):
        raise TypeError("A4 config collection must be a mapping")
    width = int(collection["width"])
    height = int(collection["height"])
    benchmark_seed = int(collection["benchmark_seed"])
    cache = load_a4_state_cache(args.cache.expanduser().resolve())
    ledger = _npz(args.ledger)
    phase_schedule = None
    if args.phase_schedule is not None:
        phase_schedule = _json(args.phase_schedule)
        selected = select_phase_balanced_bundle_indices(
            cache,
            ledger,
            phase_schedule,
            train_per_task=int(args.train_per_task),
            development_per_task=int(args.development_per_task),
        )
    else:
        selected = select_balanced_bundle_indices(
            cache,
            ledger,
            train_per_task=int(args.train_per_task),
            development_per_task=int(args.development_per_task),
        )
    if args.limit is not None:
        selected = selected[: int(args.limit)]
    base_index = replay_index_payload(cache, ledger, selected)
    if phase_schedule is not None:
        base_index["proposal_phase_name"] = phase_labels_for_bundle_indices(
            cache, ledger, selected.tolist(), phase_schedule
        )
    camera_views = _camera_views(args)
    if args.orbit_clock_panel:
        index = expand_replay_index_orbit_views(base_index)
    else:
        index = expand_replay_index_views(base_index, camera_views)
    if args.augment_gripper_actions:
        index["future_action"] = augment_gripper_action_branches(
            index["future_action"]
        )
    count = int(np.asarray(index["cache_index"]).shape[0])

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "index.npz"
    if index_path.exists():
        existing = _npz(index_path)
        for name, expected in index.items():
            if name not in existing or not np.array_equal(existing[name], expected):
                raise ValueError("existing raw replay index differs from requested selection")
    else:
        np.savez(index_path, **index)

    obs_dim = 4
    arrays = {
        "current_rgb": _allocate(output / "current_rgb.npy", (count, 3, height, width), np.uint8),
        "current_proprio": _allocate(output / "current_proprio.npy", (count, obs_dim), np.float32),
        "current_q_nominal": _allocate(output / "current_q_nominal.npy", (count, 3, 2), np.float32),
        "current_q_corrected": _allocate(output / "current_q_corrected.npy", (count, 3, 2), np.float32),
        "current_keypoints_xyz": _allocate(output / "current_keypoints_xyz.npy", (count, 3, 3), np.float32),
        "current_valid": _allocate(output / "current_valid.npy", (count, 3), np.bool_),
        "current_q_true": _allocate(output / "current_q_true.npy", (count, 3, 2), np.float32),
        "future_rgb": _allocate(output / "future_rgb.npy", (count, BRANCH_COUNT, HORIZON, 3, height, width), np.uint8),
        "future_proprio": _allocate(output / "future_proprio.npy", (count, BRANCH_COUNT, HORIZON, obs_dim), np.float32),
        "future_q_nominal": _allocate(output / "future_q_nominal.npy", (count, BRANCH_COUNT, HORIZON, 3, 2), np.float32),
        "future_q_corrected": _allocate(output / "future_q_corrected.npy", (count, BRANCH_COUNT, HORIZON, 3, 2), np.float32),
        "future_keypoints_xyz": _allocate(output / "future_keypoints_xyz.npy", (count, BRANCH_COUNT, HORIZON, 3, 3), np.float32),
        "future_valid": _allocate(output / "future_valid.npy", (count, BRANCH_COUNT, HORIZON, 3), np.bool_),
        "future_q_true": _allocate(output / "future_q_true.npy", (count, BRANCH_COUNT, HORIZON, 3, 2), np.float32),
        "completed": _allocate(output / "completed.npy", (count,), np.bool_),
    }

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
        current, branches = _collect_bundle(
            task=str(index["task_name"][output_index]),
            task_instance=int(index["task_instance"][output_index]),
            environment_seed=int(index["environment_seed"][output_index]),
            proposal_raw_step=int(index["proposal_raw_step"][output_index]),
            future_action=index["future_action"][output_index],
            width=width,
            height=height,
            benchmark_seed=benchmark_seed,
            camera_name=(
                str(index["camera_base_name"][output_index])
                if "camera_base_name" in index
                else (
                    str(index["camera_name"][output_index])
                    if "camera_name" in index
                    else None
                )
            ),
            orbit_camera_view=(
                orbit_view_from_values(
                    int(index["camera_clock_hour"][output_index]),
                    float(index["camera_elevation_degrees"][output_index]),
                    float(index["camera_radius_m"][output_index]),
                    str(index["camera_base_name"][output_index]),
                )
                if "camera_clock_hour" in index
                else None
            ),
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
        arrays["completed"][output_index] = True
        if (output_index + 1) % 8 == 0 or output_index + 1 == count:
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
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "bundle_count": count,
        "train_per_task": int(args.train_per_task),
        "development_per_task": int(args.development_per_task),
        "selection_mode": (
            "phase_balanced" if phase_schedule is not None else "episode_round_robin"
        ),
        "phase_schedule": (
            str(args.phase_schedule.expanduser().resolve())
            if args.phase_schedule is not None
            else None
        ),
        "width": width,
        "height": height,
        "tasks": list(config["tasks"]),
        "grounding_mode": str(args.grounding_mode),
        "fcpg_corrector": (
            str(args.corrector.expanduser().resolve())
            if args.grounding_mode == "fcpg"
            else None
        ),
        "camera_views": list(camera_views),
        "orbit_clock_panel": bool(args.orbit_clock_panel),
        "orbit_train_views": [view.label for view in ORBIT_TRAIN_VIEWS]
        if args.orbit_clock_panel
        else [],
        "orbit_development_views": [view.label for view in ORBIT_DEVELOPMENT_VIEWS]
        if args.orbit_clock_panel
        else [],
        "view_count": (
            len(ORBIT_TRAIN_VIEWS) + len(ORBIT_DEVELOPMENT_VIEWS)
            if args.orbit_clock_panel
            else (len(camera_views) if camera_views else 1)
        ),
        "gripper_action_augmentation": bool(args.augment_gripper_actions),
        "source_cache": str(args.cache.expanduser().resolve()),
        "source_ledger": str(args.ledger.expanduser().resolve()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if "proposal_phase_name" in base_index:
        phase_summary: dict[str, dict[str, list[str]]] = {}
        task_name = np.asarray(base_index["task_name"]).astype(str)
        split = np.asarray(base_index["split"]).astype(str)
        phase_name = np.asarray(base_index["proposal_phase_name"]).astype(str)
        for task in config["tasks"]:
            phase_summary[str(task)] = {}
            for split_name in ("train", "development"):
                mask = (task_name == str(task)) & (split == split_name)
                phase_summary[str(task)][split_name] = sorted(
                    set(phase_name[mask].tolist())
                )
        manifest["selected_phases"] = phase_summary
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
