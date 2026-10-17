#!/usr/bin/env python3
"""Audit whether selected A4b bundles cover the full expert task trajectory.

The A4b prediction gate only requires active, pre-success states.  This audit
answers a different question: does each selected H=8 bundle reach the task's
success transition when the expert is replayed from reset?
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from fcpg.lewm_a4_task_factory import make_a4_env, make_a4_expert
from fcpg.lewm_action_branch_a4 import SPLIT_SEED
from fcpg.lewm_a4b_task_identity import make_a4b_env
from fcpg.lewm_action_branch_a4_collection import a4_camera_name


RAW_HORIZON = 40
DEFAULT_LEDGER = Path(
    "outputs/lewm_action_branch_a4b/final/fresh_test_proposal_ledger.npz"
)
DEFAULT_CACHE = Path(
    "outputs/lewm_action_branch_a4b/final/fresh_test_state_cache.npz"
)
DEFAULT_OUTPUT = Path(
    "outputs/lewm_action_branch_a4b/final/trajectory_coverage_audit.json"
)
DEFAULT_RAW_OUTPUT = Path(
    "outputs/lewm_action_prefix_a4/"
    "joint_orbit12_gripper_towel_fullcycle_24train_8dev/"
    "trajectory_coverage_audit.json"
)
DEFAULT_A4C_OUTPUT = Path(
    "outputs/lewm_action_branch_a4c_v4/accepted_trajectory_coverage_audit.json"
)


def _success(info: Any) -> bool:
    if not isinstance(info, dict):
        try:
            value = info["success"]
        except (KeyError, TypeError):
            return False
    else:
        value = info.get("success", False)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return float(value) > 0.0


def load_selected_episodes(ledger_path: Path, cache_path: Path) -> dict[str, dict[str, Any]]:
    """Join accepted cache rows to their proposal metadata and group episodes."""

    with np.load(ledger_path, allow_pickle=False) as archive:
        ledger = {name: np.array(archive[name], copy=True) for name in archive.files}
    with np.load(cache_path, allow_pickle=False) as archive:
        cache = {name: np.array(archive[name], copy=True) for name in archive.files}
    accepted = np.asarray(ledger["accepted_bundle_index"], dtype=np.int64)
    episodes: dict[str, dict[str, Any]] = {}
    for cache_index in range(len(cache["proposal_id"])):
        matches = np.flatnonzero(accepted == cache_index)
        if matches.size != 1:
            raise ValueError(
                f"cache row {cache_index} has {matches.size} accepted ledger rows"
            )
        ledger_index = int(matches[0])
        episode_id = str(cache["source_episode_id"][cache_index])
        row = episodes.setdefault(
            episode_id,
            {
                "task": str(cache["task_name"][cache_index]),
                "task_instance": int(cache["task_instance_index"][cache_index]),
                "environment_seed": int(ledger["environment_seed"][ledger_index]),
                "proposal_steps": [],
            },
        )
        if row["task"] != str(cache["task_name"][cache_index]):
            raise ValueError("one source episode contains multiple tasks")
        row["proposal_steps"].append(
            int(ledger["proposal_raw_step"][ledger_index])
        )
    return episodes


def load_raw_dataset_episodes(dataset_path: Path) -> dict[str, dict[str, Any]]:
    """Load unique physical proposals from a multiview raw-data index."""

    index_path = dataset_path / "index.npz"
    with np.load(index_path, allow_pickle=False) as archive:
        required = {
            "task_name",
            "task_instance",
            "environment_seed",
            "proposal_raw_step",
            "source_episode_id",
            "split",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"raw dataset index is missing keys: {sorted(missing)}")
        arrays = {name: np.asarray(archive[name]) for name in required}

    count = len(arrays["source_episode_id"])
    if any(len(values) != count for values in arrays.values()):
        raise ValueError("raw dataset index arrays have inconsistent lengths")

    episodes: dict[str, dict[str, Any]] = {}
    for row_index in range(count):
        episode_id = str(arrays["source_episode_id"][row_index])
        task = str(arrays["task_name"][row_index])
        instance = int(arrays["task_instance"][row_index])
        environment_seed = int(arrays["environment_seed"][row_index])
        split = str(arrays["split"][row_index])
        row = episodes.setdefault(
            episode_id,
            {
                "task": task,
                "task_instance": instance,
                "environment_seed": environment_seed,
                "split": split,
                "proposal_steps": set(),
            },
        )
        identity = (task, instance, environment_seed, split)
        existing = (
            row["task"],
            row["task_instance"],
            row["environment_seed"],
            row["split"],
        )
        if identity != existing:
            raise ValueError(f"raw dataset episode identity changed for {episode_id}")
        row["proposal_steps"].add(int(arrays["proposal_raw_step"][row_index]))

    for row in episodes.values():
        row["proposal_steps"] = sorted(row["proposal_steps"])
    return episodes


def _make_a4b_episode_env(
    task: str, camera: str, width: int, height: int, raw_task_index: int
) -> Any:
    return make_a4b_env(
        task,
        camera,
        width,
        height,
        raw_task_index=raw_task_index,
    )


def _make_a4_episode_env(
    task: str, camera: str, width: int, height: int, raw_task_index: int
) -> Any:
    return make_a4_env(
        task,
        camera,
        width,
        height,
        task_instance=raw_task_index,
        benchmark_seed=SPLIT_SEED,
    )


def replay_expert_episode(
    episode: dict[str, Any],
    *,
    width: int = 32,
    height: int = 32,
    env_builder: Callable[[str, str, int, int, int], Any] = _make_a4b_episode_env,
) -> dict[str, Any]:
    """Replay one expert episode without rendering diagnostic frames."""

    task = str(episode["task"])
    env = env_builder(
        task,
        a4_camera_name(task),
        int(width),
        int(height),
        int(episode["task_instance"]),
    )
    expert = make_a4_expert(task)
    try:
        observation, _info = env.reset(seed=int(episode["environment_seed"]))
        observation = np.asarray(observation, dtype=np.float32)
        first_success: int | None = None
        stable_success: int | None = None
        streak = 0
        max_steps = int(getattr(env, "max_path_length", 500))
        last_step = 0
        for step in range(1, max_steps + 1):
            action = np.asarray(expert.get_action(observation), dtype=np.float32)
            low = np.asarray(env.action_space.low, dtype=np.float32)
            high = np.asarray(env.action_space.high, dtype=np.float32)
            if action.shape != low.shape or not np.isfinite(action).all():
                raise ValueError(f"invalid expert action for {task}")
            observation, _reward, terminated, truncated, info = env.step(
                np.clip(action, low, high)
            )
            observation = np.asarray(observation, dtype=np.float32)
            last_step = step
            if _success(info):
                if first_success is None:
                    first_success = step
                streak += 1
                if stable_success is None and streak >= 3:
                    stable_success = step
            else:
                streak = 0
            if stable_success is not None or terminated or truncated:
                break
        return {
            "first_success_step": first_success,
            "stable_success_step": stable_success,
            "trajectory_end_step": last_step,
        }
    finally:
        env.close()


def _quantiles(values: list[int]) -> list[float] | None:
    if not values:
        return None
    return [float(value) for value in np.quantile(values, [0, 0.25, 0.5, 0.75, 1])]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate bundle-level coverage without treating repeated views as samples."""

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task"])].append(row)
    result: dict[str, Any] = {}
    for task, task_rows in sorted(by_task.items()):
        count = len(task_rows)
        first = [int(row["first_success_step"]) for row in task_rows if row["first_success_step"] is not None]
        result[task] = {
            "bundle_count": count,
            "episode_count": len({str(row["episode_id"]) for row in task_rows}),
            "proposal_step_quantiles": _quantiles([int(row["proposal_step"]) for row in task_rows]),
            "first_success_quantiles": _quantiles(first),
            "trajectory_end_quantiles": _quantiles([int(row["trajectory_end_step"]) for row in task_rows]),
            "active_at_proposal_fraction": float(np.mean([row["active_at_proposal"] for row in task_rows])),
            "post_success_fraction": float(np.mean([row["post_success"] for row in task_rows])),
            "h8_reaches_first_success_fraction": float(np.mean([row["h8_reaches_first_success"] for row in task_rows])),
            "h8_reaches_stable_success_fraction": float(np.mean([row["h8_reaches_stable_success"] for row in task_rows])),
        }
    return result


def run(
    ledger_path: Path,
    cache_path: Path,
    output_path: Path,
    *,
    dataset_path: Path | None = None,
    environment: str = "a4b",
) -> dict[str, Any]:
    if dataset_path is None:
        episodes = load_selected_episodes(ledger_path, cache_path)
        if environment == "a4b":
            env_builder = _make_a4b_episode_env
            protocol = "a4b_selected_full_expert_coverage_audit_v1"
        elif environment == "a4":
            env_builder = _make_a4_episode_env
            protocol = "a4_selected_full_expert_coverage_audit_v1"
        else:
            raise ValueError("environment must be 'a4b' or 'a4'")
    else:
        episodes = load_raw_dataset_episodes(dataset_path)
        env_builder = _make_a4_episode_env
        protocol = "a4_raw_multiview_full_expert_coverage_audit_v1"
    replayed = {
        episode_id: replay_expert_episode(episode, env_builder=env_builder)
        for episode_id, episode in sorted(episodes.items())
    }
    rows: list[dict[str, Any]] = []
    for episode_id, episode in episodes.items():
        trajectory = replayed[episode_id]
        first = trajectory["first_success_step"]
        stable = trajectory["stable_success_step"]
        for proposal_step in episode["proposal_steps"]:
            rows.append(
                {
                    "episode_id": episode_id,
                    "task": episode["task"],
                    "proposal_step": int(proposal_step),
                    "first_success_step": first,
                    "stable_success_step": stable,
                    "trajectory_end_step": trajectory["trajectory_end_step"],
                    "active_at_proposal": first is None or proposal_step < first,
                    "post_success": first is not None and proposal_step >= first,
                    "h8_reaches_first_success": first is not None and proposal_step < first <= proposal_step + RAW_HORIZON,
                    "h8_reaches_stable_success": stable is not None and proposal_step < stable <= proposal_step + RAW_HORIZON,
                }
            )
    result = {
        "protocol": protocol,
        "raw_horizon_steps": RAW_HORIZON,
        "episode_count": len(episodes),
        "bundle_count": len(rows),
        "tasks": summarize(rows),
        "claim_boundary": (
            "This audits full-task trajectory coverage; it does not invalidate "
            "the active-control prediction gate."
        ),
    }
    if dataset_path is not None:
        result["dataset"] = str(dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Audit a raw multiview dataset index instead of an A4b ledger/cache.",
    )
    parser.add_argument(
        "--environment",
        choices=("a4b", "a4"),
        default="a4b",
        help="Environment family for ledger/cache replay; ignored with --dataset.",
    )
    args = parser.parse_args()
    if args.dataset is not None and args.output == DEFAULT_OUTPUT:
        args.output = DEFAULT_RAW_OUTPUT
    elif args.dataset is None and args.environment == "a4" and args.output == DEFAULT_OUTPUT:
        args.output = DEFAULT_A4C_OUTPUT
    result = run(
        args.ledger,
        args.cache,
        args.output,
        dataset_path=args.dataset,
        environment=args.environment,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
