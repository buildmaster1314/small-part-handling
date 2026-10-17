#!/usr/bin/env python3
"""Replace only Towel rows in an orbit-12 A4 replay with full-cycle windows."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.custom_tasks.towel_wipe_expert import infer_towel_wipe_phase  # noqa: E402
from fcpg.lewm_action_branch_a3 import build_diverse_action_branch_bank  # noqa: E402
from fcpg.lewm_action_branch_a4 import TASKS  # noqa: E402
from fcpg.lewm_action_branch_a4_collection import (  # noqa: E402
    _bounded_expert_action,
    _branch_seeds,
    _model_actions,
    make_a4_episode_runtime,
)
from fcpg.lewm_action_prefix_joint_data import (  # noqa: E402
    open_raw_replay_arrays,
    validate_replay_index,
)
from fcpg.lewm_orbit_camera import orbit_view_from_values  # noqa: E402
from fcpg.lewm_towel_full_cycle_replay import (  # noqa: E402
    REQUIRED_HALF_CYCLES,
    TOWEL_TASK,
    TowelProposalSpec,
    select_towel_proposal,
    towel_cycle_intervals,
    towel_split_assignments,
    validate_towel_full_cycle_specs,
)
from scripts.collect_lewm_action_prefix_joint_a4 import (  # noqa: E402
    _collect_bundle,
    _record_arrays,
    augment_gripper_action_branches,
)


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_active_control_a4.json"
DEFAULT_SOURCE = ROOT / (
    "outputs/lewm_action_prefix_a4/joint_orbit12_gripper_24train_8dev"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "joint_orbit12_gripper_towel_fullcycle_24train_8dev"
)


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-raw", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build and validate the replacement index without replaying images.",
    )
    return parser.parse_args(argv)


def _cuda0(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("Towel full-cycle replay permits only CUDA0")
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


def _clone_raw_dataset(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"replacement output already exists: {output}")
    output.mkdir(parents=True)
    subprocess.run(
        ["cp", "-a", "--reflink=auto", f"{source}/.", str(output)],
        check=True,
    )


def _trace_towel_source(
    *,
    task_instance: int,
    environment_seed: int,
    width: int,
    height: int,
    benchmark_seed: int,
) -> tuple[np.ndarray, tuple[Any, ...], np.ndarray, np.ndarray]:
    runtime = make_a4_episode_runtime(
        TOWEL_TASK,
        int(task_instance),
        int(environment_seed),
        width=int(width),
        height=int(height),
        benchmark_seed=int(benchmark_seed),
    )
    observation = np.asarray(runtime.observation, dtype=np.float32)
    phases: list[str] = []
    half_cycles: list[int] = []
    actions: list[np.ndarray] = []
    grasped: list[bool] = []
    engaged: list[bool] = []
    try:
        for _raw_step in range(int(runtime.max_raw_steps)):
            phases.append(str(infer_towel_wipe_phase(observation).value))
            half_cycles.append(int(runtime.env.wipe_half_cycles))
            grasped.append(bool(runtime.env.towel_grasped))
            engaged.append(bool(runtime.env.towel_surface_engaged))
            action = _bounded_expert_action(runtime, observation)
            actions.append(action.copy())
            observation, _reward, terminated, truncated, info = runtime.env.step(action)
            if not isinstance(info, Mapping):
                raise TypeError("Towel source info must be a mapping")
            observation = np.asarray(observation, dtype=np.float32)
            if terminated or truncated:
                break
        intervals = towel_cycle_intervals(phases, half_cycles)
        action_array = np.asarray(actions, dtype=np.float32)
        if action_array.ndim != 2 or action_array.shape[1] != 4:
            raise RuntimeError("Towel expert trace did not emit [T,4] actions")
        return (
            action_array,
            intervals,
            np.asarray(grasped, dtype=bool),
            np.asarray(engaged, dtype=bool),
        )
    finally:
        runtime.env.close()


def build_towel_full_cycle_index(
    index: Mapping[str, np.ndarray],
    *,
    contract: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], tuple[TowelProposalSpec, ...]]:
    """Replace Towel proposal steps/actions while preserving all camera rows."""

    result = {name: np.array(value, copy=True) for name, value in index.items()}
    count = validate_replay_index(result)
    task_names = np.asarray(result["task_name"]).astype(str)
    towel_rows = np.flatnonzero(task_names == TOWEL_TASK)
    if towel_rows.size == 0:
        raise ValueError("orbit replay contains no Towel rows")
    physical = np.asarray(result["physical_bundle_index"], dtype=np.int64)
    splits = np.asarray(result["split"]).astype(str)
    task_index = TASKS.index(TOWEL_TASK)
    collection = contract.get("collection")
    if not isinstance(collection, Mapping):
        raise TypeError("A4 contract collection must be a mapping")
    width = int(collection["width"])
    height = int(collection["height"])
    benchmark_seed = int(collection["benchmark_seed"])

    metadata: dict[str, np.ndarray] = {
        "towel_full_cycle_replacement": np.zeros(count, dtype=bool),
        "towel_cycle_index": np.full(count, -1, dtype=np.int64),
        "towel_cycle_fraction": np.full(count, np.nan, dtype=np.float32),
        "towel_cycle_start_raw_step": np.full(count, -1, dtype=np.int64),
        "towel_cycle_end_raw_step": np.full(count, -1, dtype=np.int64),
        "towel_cycle_direction": np.full(count, "", dtype="<U5"),
        "legacy_cache_index": np.asarray(result["cache_index"], dtype=np.int64).copy(),
    }
    traces: dict[tuple[int, int], tuple[np.ndarray, tuple[Any, ...], np.ndarray, np.ndarray]] = {}
    specs: list[TowelProposalSpec] = []
    for split, expected_count in (("train", 24), ("development", 8)):
        split_rows = towel_rows[splits[towel_rows] == split]
        physical_ids = []
        seen: set[int] = set()
        for row in split_rows.tolist():
            bundle = int(physical[row])
            if bundle not in seen:
                seen.add(bundle)
                physical_ids.append(bundle)
        assignments = towel_split_assignments(split, len(physical_ids))
        if len(physical_ids) != expected_count:
            raise ValueError(
                f"Towel {split} physical bundle count drifted from {expected_count}"
            )
        for ordinal, (bundle, assignment) in enumerate(
            zip(physical_ids, assignments, strict=True)
        ):
            rows = np.flatnonzero((physical == bundle) & (task_names == TOWEL_TASK))
            if rows.size == 0 or not bool(np.all(splits[rows] == split)):
                raise RuntimeError("Towel physical bundle/view grouping is inconsistent")
            representative = int(rows[0])
            instance = int(result["task_instance"][representative])
            environment_seed = int(result["environment_seed"][representative])
            trace_key = (instance, environment_seed)
            if trace_key not in traces:
                traces[trace_key] = _trace_towel_source(
                    task_instance=instance,
                    environment_seed=environment_seed,
                    width=width,
                    height=height,
                    benchmark_seed=benchmark_seed,
                )
            actions, intervals, grasped, engaged = traces[trace_key]
            cycle_index, fraction = assignment
            spec = select_towel_proposal(
                intervals[cycle_index],
                split=split,
                bundle_ordinal=ordinal,
                fraction=fraction,
            )
            proposal = int(spec.proposal_raw_step)
            if proposal + 40 > actions.shape[0]:
                raise RuntimeError("Towel proposal lacks a complete H=8 expert center")
            if not bool(grasped[proposal]) or not bool(engaged[proposal]):
                raise RuntimeError("Towel full-cycle proposal is not grasped on the surface")
            direction_seed, permutation_seed = _branch_seeds(
                contract, task_index, instance, proposal
            )
            runtime = make_a4_episode_runtime(
                TOWEL_TASK,
                instance,
                environment_seed,
                width=width,
                height=height,
                benchmark_seed=benchmark_seed,
            )
            try:
                bank = build_diverse_action_branch_bank(
                    np.ascontiguousarray(actions[proposal : proposal + 40]),
                    direction_seed=direction_seed,
                    permutation_seed=permutation_seed,
                    action_low=runtime.env.action_space.low,
                    action_high=runtime.env.action_space.high,
                )
            finally:
                runtime.env.close()
            future_action = np.stack(
                [_model_actions(branch) for branch in bank.actions]
            ).astype(np.float32, copy=False)
            future_action = augment_gripper_action_branches(future_action[None])[0]
            proposal_id = (
                f"{result['source_episode_id'][representative]}:"
                f"fullcycle-{split}-{ordinal:02d}-c{cycle_index}-t{proposal:03d}"
            )
            result["cache_index"][rows] = -1 - int(bundle)
            result["proposal_id"][rows] = proposal_id
            result["proposal_raw_step"][rows] = proposal
            result["future_action"][rows] = future_action
            metadata["towel_full_cycle_replacement"][rows] = True
            metadata["towel_cycle_index"][rows] = int(cycle_index)
            metadata["towel_cycle_fraction"][rows] = float(fraction)
            metadata["towel_cycle_start_raw_step"][rows] = int(
                spec.interval_start_raw_step
            )
            metadata["towel_cycle_end_raw_step"][rows] = int(
                spec.interval_end_raw_step
            )
            metadata["towel_cycle_direction"][rows] = str(spec.direction)
            specs.append(spec)
    validate_towel_full_cycle_specs(specs)
    result.update(metadata)
    validate_replay_index(result)
    if not np.array_equal(
        np.asarray(index["future_action"])[task_names != TOWEL_TASK],
        result["future_action"][task_names != TOWEL_TASK],
    ):
        raise RuntimeError("Towel replacement modified a non-Towel action row")
    return result, tuple(specs)


def _coverage_payload(specs: Sequence[TowelProposalSpec]) -> dict[str, Any]:
    rows = tuple(specs)
    return {
        "status": "coverage_go",
        "proposal_count": len(rows),
        "splits": {
            split: {
                "count": sum(row.split == split for row in rows),
                "cycles": {
                    str(cycle): sum(
                        row.split == split and row.cycle_index == cycle for row in rows
                    )
                    for cycle in range(REQUIRED_HALF_CYCLES)
                },
                "proposal_raw_step_min": min(
                    row.proposal_raw_step for row in rows if row.split == split
                ),
                "proposal_raw_step_max": max(
                    row.proposal_raw_step for row in rows if row.split == split
                ),
                "directions": sorted(
                    {row.direction for row in rows if row.split == split}
                ),
            }
            for split in ("train", "development")
        },
        "specs": [
            {
                "split": row.split,
                "bundle_ordinal": row.bundle_ordinal,
                "cycle_index": row.cycle_index,
                "fraction": row.fraction,
                "interval_start_raw_step": row.interval_start_raw_step,
                "interval_end_raw_step": row.interval_end_raw_step,
                "proposal_raw_step": row.proposal_raw_step,
                "direction": row.direction,
            }
            for row in rows
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser(argv)
    source = args.source_raw.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    contract = _json(args.config)
    source_index = _npz(source / "index.npz")
    replacement_index, specs = build_towel_full_cycle_index(
        source_index, contract=contract
    )
    coverage = _coverage_payload(specs)
    print(json.dumps(coverage, sort_keys=True), flush=True)
    if args.prepare_only:
        return 0

    device = _cuda0(args.device)
    torch.cuda.set_device(device)
    _clone_raw_dataset(source, output)
    index_path = output / "index.npz"
    with open(index_path, "wb") as handle:
        np.savez(handle, **replacement_index)
    index, arrays = open_raw_replay_arrays(output, mode="r+")
    towel_rows = np.flatnonzero(
        np.asarray(index["task_name"]).astype(str) == TOWEL_TASK
    )
    arrays["completed"][towel_rows] = False
    arrays["completed"].flush()

    collection = contract["collection"]
    width = int(collection["width"])
    height = int(collection["height"])
    benchmark_seed = int(collection["benchmark_seed"])
    started = time.perf_counter()
    for completed_index, row in enumerate(towel_rows.tolist(), start=1):
        current, branches = _collect_bundle(
            task=TOWEL_TASK,
            task_instance=int(index["task_instance"][row]),
            environment_seed=int(index["environment_seed"][row]),
            proposal_raw_step=int(index["proposal_raw_step"][row]),
            future_action=index["future_action"][row],
            width=width,
            height=height,
            benchmark_seed=benchmark_seed,
            camera_name=str(index["camera_base_name"][row]),
            orbit_camera_view=orbit_view_from_values(
                int(index["camera_clock_hour"][row]),
                float(index["camera_elevation_degrees"][row]),
                float(index["camera_radius_m"][row]),
                str(index["camera_base_name"][row]),
            ),
        )
        values = _record_arrays(
            current,
            branches,
            corrector=None,
            device=device,
            grounding_mode="nominal",
        )
        for name, value in values.items():
            arrays[name][row] = value
        arrays["completed"][row] = True
        if completed_index % 8 == 0 or completed_index == towel_rows.size:
            for array in arrays.values():
                if hasattr(array, "flush"):
                    array.flush()
            print(
                json.dumps(
                    {
                        "completed_towel_rows": completed_index,
                        "total_towel_rows": int(towel_rows.size),
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    manifest = _json(output / "manifest.json")
    manifest.update(
        {
            "source_raw_dataset": str(source),
            "towel_full_cycle_replacement": True,
            "towel_replaced_view_rows": int(towel_rows.size),
            "towel_coverage": coverage,
            "elapsed_towel_replay_seconds": time.perf_counter() - started,
        }
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
