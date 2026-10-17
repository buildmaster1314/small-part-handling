"""Physical replay diagnostics for the A3 action-branch support Gate."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from fcpg.lewm_action_branch_a1 import CANDIDATE_COUNT, validate_a1_arrays
from fcpg.lewm_action_branch_a3_collection import (
    TASKS,
    validate_a3_cache_conformance,
    validate_a3_contract,
)
from fcpg.lewm_action_branch_collection import (
    ACTION_DIM,
    FRAME_SKIP,
    EpisodeRuntime,
    capture_branch_snapshot,
    restore_branch_snapshot,
)
from fcpg.lewm_counterfactual_v2_runtime import _close_episode_env_safely


PHYSICAL_KEYS = frozenset(
    {
        "task_index",
        "task_instance_index",
        "source_episode_id",
        "hazard_id",
        "split",
        "source_tcp",
        "source_object_position",
        "tcp",
        "object_position",
    }
)


def read_generic_physical_state(env: Any) -> tuple[np.ndarray, np.ndarray]:
    """Read task-independent robot and manipulated-object positions."""

    tcp = np.asarray(env.tcp_center, dtype=np.float32)
    object_position = np.asarray(env._get_pos_objects(), dtype=np.float32)
    if tcp.shape != (3,) or object_position.shape != (3,):
        raise ValueError("A3 physical state requires 3D TCP and object positions")
    if not np.isfinite(tcp).all() or not np.isfinite(object_position).all():
        raise ValueError("A3 physical state must be finite")
    return tcp.copy(), object_position.copy()


def replay_physical_action_branches(
    runtime: EpisodeRuntime,
    snapshot: Any,
    branch_actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay exact cached actions and retain physical H1-H8 endpoints."""

    actions = np.asarray(branch_actions, dtype=np.float32)
    if actions.ndim != 3 or actions.shape[0] != CANDIDATE_COUNT:
        raise ValueError("A3 branch_actions must have shape [8,raw_steps,4]")
    if actions.shape[2] != ACTION_DIM or actions.shape[1] % FRAME_SKIP:
        raise ValueError("A3 branch action dimensions are invalid")
    low = np.asarray(runtime.env.action_space.low, dtype=np.float32)
    high = np.asarray(runtime.env.action_space.high, dtype=np.float32)
    if not np.array_equal(actions, np.clip(actions, low, high)):
        raise ValueError("A3 physical replay refuses actions requiring clipping")
    horizon = actions.shape[1] // FRAME_SKIP
    tcp_rows: list[list[np.ndarray]] = []
    object_rows: list[list[np.ndarray]] = []
    for branch in actions:
        restore_branch_snapshot(runtime.env, snapshot)
        tcp_endpoints: list[np.ndarray] = []
        object_endpoints: list[np.ndarray] = []
        for raw_index, action in enumerate(branch, start=1):
            _, _, terminated, truncated, _ = runtime.env.step(action)
            if raw_index % FRAME_SKIP == 0:
                tcp, object_position = read_generic_physical_state(runtime.env)
                tcp_endpoints.append(tcp)
                object_endpoints.append(object_position)
            if (terminated or truncated) and raw_index < branch.shape[0]:
                raise RuntimeError("A3 physical branch ended before H8")
        if len(tcp_endpoints) != horizon:
            raise RuntimeError("A3 physical branch endpoint count is incomplete")
        tcp_rows.append(tcp_endpoints)
        object_rows.append(object_endpoints)
    return (
        np.asarray(tcp_rows, dtype=np.float32),
        np.asarray(object_rows, dtype=np.float32),
    )


def validate_reconstructed_source_actions(
    *,
    history_action: np.ndarray,
    center_action: np.ndarray,
    hazard_step: int,
    source_actions: np.ndarray,
) -> None:
    """Require reconstructed expert actions to match the cached source exactly."""

    history = np.asarray(history_action, dtype=np.float32).reshape(-1, ACTION_DIM)
    center = np.asarray(center_action, dtype=np.float32).reshape(-1, ACTION_DIM)
    source = np.asarray(source_actions, dtype=np.float32)
    raw_hazard = int(hazard_step) * FRAME_SKIP
    if history.shape != (2 * FRAME_SKIP, ACTION_DIM):
        raise ValueError("A3 cached history_action must contain ten raw actions")
    if center.shape != (8 * FRAME_SKIP, ACTION_DIM):
        raise ValueError("A3 cached center_action must contain forty raw actions")
    if source.ndim != 2 or source.shape[1] != ACTION_DIM:
        raise ValueError("A3 reconstructed source actions must have shape [T,4]")
    if raw_hazard < history.shape[0] or raw_hazard + center.shape[0] > source.shape[0]:
        raise ValueError("A3 reconstructed source trajectory does not cover the cache")
    if not np.array_equal(history, source[raw_hazard - history.shape[0] : raw_hazard]):
        raise ValueError("A3 reconstructed history actions do not match the cache")
    if not np.array_equal(center, source[raw_hazard : raw_hazard + center.shape[0]]):
        raise ValueError("A3 reconstructed center actions do not match the cache")


def replay_a3_physical_rows(
    latent_arrays: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
    *,
    runtime_factory: Callable[[str, int, int], EpisodeRuntime],
) -> dict[str, np.ndarray]:
    """Reconstruct source snapshots and replay every cached A3 branch."""

    validate_a1_arrays(latent_arrays)
    tasks = validate_a3_contract(contract)
    validate_a3_cache_conformance(latent_arrays, contract)
    task_index = np.asarray(latent_arrays["task_index"], dtype=np.int64)
    task_instance = np.asarray(
        latent_arrays["task_instance_index"], dtype=np.int64
    )
    hazard = np.asarray(latent_arrays["hazard_id"], dtype=np.int64)
    split = np.asarray(latent_arrays["split"]).astype(str)
    source_episode = np.asarray(latent_arrays["source_episode_id"]).astype(str)
    history_action = np.asarray(latent_arrays["history_action"], dtype=np.float32)
    center_action = np.asarray(latent_arrays["center_action"], dtype=np.float32)
    future_action = np.asarray(latent_arrays["future_action"], dtype=np.float32)
    if np.any(split == "test"):
        raise ValueError("A3 zero-training physical replay refuses test rows")
    expected_task_indices = set(range(len(tasks)))
    if set(task_index.tolist()) - expected_task_indices:
        raise ValueError("A3 latent cache contains an unknown task index")
    if future_action.ndim != 4 or future_action.shape[1:3] != (8, 8):
        raise ValueError("A3 future_action must have shape [N,8,8,20]")

    count = int(task_index.size)
    episode_to_instance: dict[tuple[int, str], int] = {}
    instance_to_episode: dict[tuple[int, int], str] = {}
    for row in range(task_index.size):
        episode_key = (int(task_index[row]), str(source_episode[row]))
        instance_key = (int(task_index[row]), int(task_instance[row]))
        previous_instance = episode_to_instance.setdefault(
            episode_key, int(task_instance[row])
        )
        previous_episode = instance_to_episode.setdefault(
            instance_key, str(source_episode[row])
        )
        if (
            previous_instance != int(task_instance[row])
            or previous_episode != str(source_episode[row])
        ):
            raise ValueError("A3 source episode and task instance are not one-to-one")
    bundle_identity = np.stack((task_index, task_instance, hazard), axis=1)
    if np.unique(bundle_identity, axis=0).shape[0] != count:
        raise ValueError("A3 physical replay contains duplicate episode hazards")

    source_tcp = np.empty((count, 3), dtype=np.float32)
    source_object = np.empty((count, 3), dtype=np.float32)
    tcp = np.empty((count, 8, 8, 3), dtype=np.float32)
    object_position = np.empty((count, 8, 8, 3), dtype=np.float32)
    collection = contract.get("collection")
    if not isinstance(collection, Mapping):
        raise TypeError("collection must be a mapping")
    environment_seed_base = int(collection["environment_seed_base"])

    for current_task_index, task in enumerate(tasks):
        task_rows = np.flatnonzero(task_index == current_task_index)
        for instance in np.unique(task_instance[task_rows]):
            selected = task_rows[task_instance[task_rows] == int(instance)]
            if selected.size == 0:
                continue
            row_splits = set(split[selected].tolist())
            if len(row_splits) != 1:
                raise ValueError("one A3 physical episode crosses splits")
            row_hazards = hazard[selected]
            if np.unique(row_hazards).size != selected.size:
                raise ValueError("one A3 episode contains duplicate hazard rows")
            environment_seed = (
                environment_seed_base
                + (1_000_000 if row_splits == {"smoke"} else 0)
                + current_task_index * 10_000
                + int(instance)
            )
            runtime = runtime_factory(task, int(instance), environment_seed)
            try:
                observation = np.asarray(runtime.observation, dtype=np.float32)
                by_raw_step = {
                    int(hazard[row]) * FRAME_SKIP: int(row) for row in selected
                }
                snapshots: dict[int, Any] = {}
                final_raw_step = max(by_raw_step) + 8 * FRAME_SKIP
                reconstructed_actions: list[np.ndarray] = []
                for raw_step in range(final_raw_step + 1):
                    if raw_step in by_raw_step:
                        row = by_raw_step[raw_step]
                        snapshots[row] = capture_branch_snapshot(
                            runtime.env, include_process_rng=True
                        )
                        source_tcp[row], source_object[row] = (
                            read_generic_physical_state(runtime.env)
                        )
                    if raw_step == final_raw_step:
                        break
                    action = np.asarray(
                        runtime.expert.get_action(observation), dtype=np.float32
                    )
                    action = np.clip(
                        action,
                        runtime.env.action_space.low,
                        runtime.env.action_space.high,
                    ).astype(np.float32)
                    reconstructed_actions.append(action.copy())
                    observation, _, terminated, truncated, _ = runtime.env.step(action)
                    observation = np.asarray(observation, dtype=np.float32)
                    if (terminated or truncated) and raw_step + 1 < final_raw_step:
                        raise RuntimeError("A3 source ended before a selected hazard")

                source_actions = np.asarray(reconstructed_actions, dtype=np.float32)
                for row in selected:
                    if not source_episode[row].startswith(f"{task}:"):
                        raise ValueError("A3 task index and source identity disagree")
                    validate_reconstructed_source_actions(
                        history_action=history_action[row],
                        center_action=center_action[row],
                        hazard_step=int(hazard[row]),
                        source_actions=source_actions,
                    )
                    raw_actions = future_action[row].reshape(8, -1, ACTION_DIM)
                    tcp[row], object_position[row] = replay_physical_action_branches(
                        runtime, snapshots[int(row)], raw_actions
                    )
            finally:
                _close_episode_env_safely(runtime.env)

    physical = {
        "task_index": task_index.copy(),
        "task_instance_index": task_instance.copy(),
        "source_episode_id": source_episode.copy(),
        "hazard_id": hazard.copy(),
        "split": split.copy(),
        "source_tcp": source_tcp,
        "source_object_position": source_object,
        "tcp": tcp,
        "object_position": object_position,
    }
    validate_a3_physical_arrays(physical, latent_arrays=latent_arrays)
    return physical


def validate_a3_physical_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    latent_arrays: Mapping[str, np.ndarray] | None = None,
) -> None:
    if set(arrays) != PHYSICAL_KEYS:
        raise ValueError(f"A3 physical keys must be exactly {sorted(PHYSICAL_KEYS)}")
    count = int(np.asarray(arrays["task_index"]).size)
    expected = {
        "task_index": (count,),
        "task_instance_index": (count,),
        "source_episode_id": (count,),
        "hazard_id": (count,),
        "split": (count,),
        "source_tcp": (count, 3),
        "source_object_position": (count, 3),
        "tcp": (count, 8, 8, 3),
        "object_position": (count, 8, 8, 3),
    }
    for name, shape in expected.items():
        value = np.asarray(arrays[name])
        if value.shape != shape:
            raise ValueError(f"A3 physical {name} must have shape {shape}")
        if value.dtype.kind in "fiu" and not np.isfinite(value).all():
            raise ValueError(f"A3 physical {name} must be finite")
    if latent_arrays is not None:
        for name in (
            "task_index",
            "task_instance_index",
            "source_episode_id",
            "hazard_id",
            "split",
        ):
            if not np.array_equal(np.asarray(arrays[name]), np.asarray(latent_arrays[name])):
                raise ValueError(f"A3 physical identity {name} is not row-aligned")


__all__ = [
    "PHYSICAL_KEYS",
    "read_generic_physical_state",
    "replay_a3_physical_rows",
    "replay_physical_action_branches",
    "validate_reconstructed_source_actions",
    "validate_a3_physical_arrays",
]
