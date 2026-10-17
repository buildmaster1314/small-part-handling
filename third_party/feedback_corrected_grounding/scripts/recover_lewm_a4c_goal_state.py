#!/usr/bin/env python3
"""Recover explicit world goals for the selected seven-task A4c control rows."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_a4_task_signals import extract_a4_task_signals  # noqa: E402
from fcpg.lewm_a4c_goal_state import (  # noqa: E402
    RecoveredGoalState,
    build_a4c_goal_state_index,
    proposal_identity,
    selected_proposal_identities,
)
from fcpg.lewm_action_branch_a4_collection import (  # noqa: E402
    _bounded_expert_action,
    make_a4_episode_runtime,
)


DEFAULT_RAW_ROOT = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "joint_orbit12_gripper_phaseaware_a4c_v4_48train_24dev"
)
DEFAULT_A4C_ROOT = ROOT / "outputs/lewm_action_branch_a4c_v4"
DEFAULT_CONTROL_INDEX = ROOT / (
    "outputs/lewm_action_prefix_a4/a4c_control_index_v1/index.npz"
)
DEFAULT_OUTPUT = ROOT / "outputs/lewm_action_prefix_a4/a4c_goal_state_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--a4c-root", type=Path, default=DEFAULT_A4C_ROOT)
    parser.add_argument("--control-index", type=Path, default=DEFAULT_CONTROL_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-every", type=int, default=10)
    return parser


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.expanduser().resolve(), allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _replay_selected(
    identities: tuple[tuple[str, int, int, int], ...],
    *,
    width: int,
    height: int,
    benchmark_seed: int,
    log_every: int,
) -> dict[tuple[str, int, int, int], RecoveredGoalState]:
    episodes: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for task, instance, seed, step in identities:
        episodes[(task, instance, seed)].append(step)
    recovered: dict[tuple[str, int, int, int], RecoveredGoalState] = {}
    for episode_index, ((task, instance, seed), selected_steps) in enumerate(
        sorted(episodes.items()), start=1
    ):
        wanted = set(selected_steps)
        maximum_step = max(wanted)
        runtime = make_a4_episode_runtime(
            task,
            instance,
            seed,
            width=width,
            height=height,
            benchmark_seed=benchmark_seed,
        )
        observation = np.asarray(runtime.observation, dtype=np.float32)
        info: dict[str, Any] = dict(runtime.reset_info)
        try:
            for raw_step in range(maximum_step + 1):
                if raw_step in wanted:
                    signals = extract_a4_task_signals(
                        task, runtime.env, observation, info
                    )
                    diagnostics = signals.diagnostics
                    tcp = np.asarray(runtime.env.tcp_center, dtype=np.float32)
                    recovered[proposal_identity(task, instance, seed, raw_step)] = (
                        RecoveredGoalState(
                            current_proprio=np.asarray(
                                signals.deployment.proprio, dtype=np.float32
                            ).copy(),
                            object_pose=np.asarray(
                                diagnostics.object_pose, dtype=np.float32
                            ).copy(),
                            reference_pose=np.asarray(
                                diagnostics.reference_pose, dtype=np.float32
                            ).copy(),
                            tcp_position=tcp.copy(),
                            task_progress=float(diagnostics.task_progress),
                            goal_relative_progress=float(
                                diagnostics.goal_relative_progress
                            ),
                            contact_or_grasp_state=bool(
                                diagnostics.contact_or_grasp_state
                            ),
                            success=bool(diagnostics.success),
                            active_stage=bool(diagnostics.active_stage),
                        )
                    )
                if raw_step == maximum_step:
                    break
                action = _bounded_expert_action(runtime, observation)
                observation, _reward, terminated, truncated, next_info = (
                    runtime.env.step(action)
                )
                if terminated or truncated:
                    raise RuntimeError(
                        f"{task} episode ended before selected step {maximum_step}"
                    )
                if not isinstance(next_info, dict):
                    next_info = dict(next_info)
                info = next_info
                observation = np.asarray(observation, dtype=np.float32)
        finally:
            runtime.env.close()
        if episode_index % log_every == 0 or episode_index == len(episodes):
            print(
                json.dumps(
                    {
                        "episodes_replayed": episode_index,
                        "episodes_total": len(episodes),
                        "states_recovered": len(recovered),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return recovered


def main() -> int:
    args = _parser().parse_args()
    if int(args.log_every) <= 0:
        raise ValueError("--log-every must be positive")
    raw_root = args.raw_root.expanduser().resolve()
    a4c_root = args.a4c_root.expanduser().resolve()
    control = _npz(args.control_index)
    raw = _npz(raw_root / "index.npz")
    manifest = json.loads((a4c_root / "phase_schedule.json").read_text(encoding="utf-8"))
    collection = manifest["collection"]
    identities = selected_proposal_identities(control, raw)
    recovered = _replay_selected(
        identities,
        width=int(collection["width"]),
        height=int(collection["height"]),
        benchmark_seed=int(collection["benchmark_seed"]),
        log_every=int(args.log_every),
    )
    payload, summary = build_a4c_goal_state_index(
        control,
        raw,
        np.load(raw_root / "current_proprio.npy", mmap_mode="r"),
        np.load(raw_root / "current_keypoints_xyz.npy", mmap_mode="r"),
        recovered,
    )
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "index.npz", **payload)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **summary}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
