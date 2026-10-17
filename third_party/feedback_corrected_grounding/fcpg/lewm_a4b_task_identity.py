"""Fresh task construction and reset-identity auditing for A4b."""

from __future__ import annotations

import hashlib
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from fcpg import metaworld_utils
from fcpg.custom_tasks.drawer_retrieve import SawyerDrawerRetrieveEnvV0
from fcpg.custom_tasks.towel_wipe import SawyerTowelWipeEnvV0
from fcpg.lewm_a4_task_factory import (
    CUSTOM_TASKS,
    DRAWER_RETRIEVE_TASK,
    TOWEL_WIPE_TASK,
)
from fcpg.lewm_action_branch_a4 import FORMAL_TASK_INSTANCES, TASKS, assign_a4_splits
from fcpg.lewm_action_branch_a4_collection import (
    A4EpisodeRuntime,
    a4_camera_name,
)
from fcpg.lewm_action_branch_a4b import (
    BENCHMARK_SEED,
    CUSTOM_TASK_PARAMETER_SEED,
    RAW_TASK_INDICES,
)


RESET_DIGEST_FIELDS = ("qpos", "qvel", "goal", "task_data")
LEGACY_IDENTITY_SCOPES = (
    "a4_v2_formal_train",
    "a4_v2_formal_development",
    "a4_v2_attempted_test",
)


def _fresh_offsets(task: str, raw_task_index: int, size: int) -> np.ndarray:
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                CUSTOM_TASK_PARAMETER_SEED,
                zlib.crc32(task.encode("utf-8")),
                int(raw_task_index),
            ]
        )
    )
    values = rng.uniform(-1.0, 1.0, size=size)
    signs = np.where(values < 0.0, -1.0, 1.0)
    return signs * (0.001 + 0.003 * np.abs(values))


class A4bDrawerRetrieveEnvV1(SawyerDrawerRetrieveEnvV0):
    """Drawer task with protocol-seeded continuous reset/goal offsets."""

    @classmethod
    def task_instance_reset_parameters(
        cls, task_instance: int
    ) -> tuple[np.ndarray, np.ndarray]:
        block, target = super().task_instance_reset_parameters(task_instance)
        offsets = _fresh_offsets(DRAWER_RETRIEVE_TASK, task_instance, 4)
        fresh_block = block.copy()
        fresh_target = target.copy()
        fresh_block[:2] += offsets[:2]
        fresh_target[:2] += offsets[2:]
        return fresh_block, fresh_target


class A4bTowelWipeEnvV1(SawyerTowelWipeEnvV0):
    """Towel task with protocol-seeded continuous endpoint offsets."""

    @classmethod
    def task_instance_endpoints(
        cls, task_instance: int
    ) -> tuple[np.ndarray, np.ndarray]:
        left, right = super().task_instance_endpoints(task_instance)
        span_offset, lane_offset = _fresh_offsets(
            TOWEL_WIPE_TASK, task_instance, 2
        )
        fresh_left = left.copy()
        fresh_right = right.copy()
        fresh_left[0] -= abs(float(span_offset))
        fresh_right[0] += abs(float(span_offset))
        fresh_left[1] += lane_offset
        fresh_right[1] += lane_offset
        return fresh_left, fresh_right


_A4B_CUSTOM_TASKS = {
    DRAWER_RETRIEVE_TASK: A4bDrawerRetrieveEnvV1,
    TOWEL_WIPE_TASK: A4bTowelWipeEnvV1,
}


@dataclass(frozen=True)
class ResetPhysicalIdentity:
    task_name: str
    raw_task_index: int
    benchmark_seed: int
    environment_seed: int
    camera_name: str
    digest: str


def make_a4b_env(
    task_name: str,
    camera_name: str,
    width: int,
    height: int,
    *,
    raw_task_index: int,
) -> Any:
    """Create one task in the fresh A4b benchmark/parameter namespace."""

    if task_name not in TASKS:
        raise ValueError(f"unknown A4b task {task_name!r}")
    if int(raw_task_index) not in RAW_TASK_INDICES:
        raise ValueError("A4b raw task index must be in 0..47")
    if task_name in CUSTOM_TASKS:
        return _A4B_CUSTOM_TASKS[task_name](
            task_instance=int(raw_task_index),
            render_mode="rgb_array",
            camera_name=camera_name,
            width=int(width),
            height=int(height),
            seed=BENCHMARK_SEED,
        )
    return metaworld_utils.make_env(
        task_name,
        camera_name,
        int(width),
        int(height),
        task_index=int(raw_task_index),
        benchmark_seed=BENCHMARK_SEED,
    )


def _goal(env: Any) -> np.ndarray:
    for name in ("_target_pos", "goal"):
        value = getattr(env, name, None)
        if value is not None:
            result = np.asarray(value, dtype=np.float64).reshape(-1)
            if result.size:
                return result
    raise RuntimeError("A4b reset identity requires an explicit environment goal")


def _task_data(env: Any) -> np.ndarray:
    values: list[np.ndarray] = []
    for name in (
        "_last_rand_vec",
        "obj_init_pos",
        "_left_endpoint",
        "_right_endpoint",
    ):
        value = getattr(env, name, None)
        if value is not None:
            array = np.asarray(value, dtype=np.float64).reshape(-1)
            if array.size:
                values.append(array)
    if not values:
        raise RuntimeError("A4b reset identity requires task reset data")
    return np.concatenate(values)


def reset_physical_state_goal_digest(env: Any) -> str:
    """Hash physical reset values only; protocol names/seeds are excluded."""

    fields = {
        "qpos": np.asarray(env.data.qpos, dtype=np.float64).reshape(-1),
        "qvel": np.asarray(env.data.qvel, dtype=np.float64).reshape(-1),
        "goal": _goal(env),
        "task_data": _task_data(env),
    }
    digest = hashlib.sha256()
    for name in RESET_DIGEST_FIELDS:
        value = np.ascontiguousarray(fields[name], dtype="<f8")
        if not value.size or not np.isfinite(value).all():
            raise ValueError(f"A4b reset identity field {name} must be finite/non-empty")
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def capture_reset_physical_identity(
    env: Any,
    *,
    task_name: str,
    raw_task_index: int,
    benchmark_seed: int,
    environment_seed: int,
    camera_name: str,
) -> ResetPhysicalIdentity:
    return ResetPhysicalIdentity(
        task_name=task_name,
        raw_task_index=int(raw_task_index),
        benchmark_seed=int(benchmark_seed),
        environment_seed=int(environment_seed),
        camera_name=camera_name,
        digest=reset_physical_state_goal_digest(env),
    )


@dataclass
class A4bEpisodeRuntime(A4EpisodeRuntime):
    reset_identity: ResetPhysicalIdentity


def make_a4b_episode_runtime(
    task: str,
    raw_task_index: int,
    environment_seed: int,
    *,
    width: int,
    height: int,
) -> A4bEpisodeRuntime:
    """Build/reset A4b runtime and bind its fresh physical identity."""

    from fcpg.lewm_a4_task_factory import make_a4_expert
    from fcpg.lewm_counterfactual_runtime import _collect_frame_record
    from fcpg.mujoco_camera import (
        camera_extrinsic_cv_from_mujoco,
        camera_intrinsic_from_fovy,
        get_camera_id,
    )

    camera = a4_camera_name(task)
    env = make_a4b_env(
        task,
        camera,
        int(width),
        int(height),
        raw_task_index=int(raw_task_index),
    )
    try:
        expert = make_a4_expert(task)
        observation, reset_info = env.reset(seed=int(environment_seed))
        if not isinstance(reset_info, Mapping):
            raise TypeError("A4b environment reset info must be a mapping")
        identity = capture_reset_physical_identity(
            env,
            task_name=task,
            raw_task_index=int(raw_task_index),
            benchmark_seed=BENCHMARK_SEED,
            environment_seed=int(environment_seed),
            camera_name=camera,
        )
        camera_id = get_camera_id(env.model, camera)
        intrinsic = camera_intrinsic_from_fovy(
            int(width), int(height), float(env.model.cam_fovy[camera_id])
        )
        nominal_extrinsic = camera_extrinsic_cv_from_mujoco(env.data, camera_id)

        def record_current(current_observation: np.ndarray) -> Any:
            return _collect_frame_record(
                env,
                current_observation,
                camera_id=camera_id,
                intrinsic=intrinsic,
                nominal_extrinsic=nominal_extrinsic,
                width=int(width),
                height=int(height),
            )

        return A4bEpisodeRuntime(
            env=env,
            expert=expert,
            observation=np.asarray(observation, dtype=np.float32),
            reset_info=dict(reset_info),
            record_current=record_current,
            max_raw_steps=int(getattr(env, "max_path_length", 500)),
            camera_name=camera,
            camera_intrinsic=np.asarray(intrinsic, dtype=np.float32).copy(),
            camera_extrinsic_cam_from_world=np.asarray(
                nominal_extrinsic, dtype=np.float32
            ).copy(),
            reset_identity=identity,
        )
    except Exception:
        env.close()
        raise


def collect_legacy_a4_reset_digests(
    source_contract: Mapping[str, Any],
    *,
    runtime_factory: Callable[[str, int, int], A4EpisodeRuntime],
) -> dict[str, list[dict[str, Any]]]:
    """Reconstruct all old train/dev/attempted-test reset identities."""

    assignments = assign_a4_splits(TASKS, source_contract)
    collection = source_contract["collection"]
    seed_base = int(collection["environment_seed_base"])
    benchmark_seed = int(collection["benchmark_seed"])
    result = {
        "a4_v2_formal_train": [],
        "a4_v2_formal_development": [],
        "a4_v2_attempted_test": [],
    }
    scope_by_split = {
        "train": "a4_v2_formal_train",
        "development": "a4_v2_formal_development",
        "test": "a4_v2_attempted_test",
    }
    for task_index, task in enumerate(TASKS):
        for raw_index in FORMAL_TASK_INSTANCES:
            environment_seed = (
                seed_base + task_index * len(FORMAL_TASK_INSTANCES) + raw_index
            )
            runtime = runtime_factory(task, raw_index, environment_seed)
            try:
                identity = capture_reset_physical_identity(
                    runtime.env,
                    task_name=task,
                    raw_task_index=raw_index,
                    benchmark_seed=benchmark_seed,
                    environment_seed=environment_seed,
                    camera_name=a4_camera_name(task),
                )
            finally:
                runtime.env.close()
            scope = scope_by_split[assignments[(task, raw_index)]]
            result[scope].append(
                {
                    "task_name": task,
                    "raw_task_index": raw_index,
                    "digest": identity.digest,
                }
            )
    return result


def legacy_digest_set(
    inventory: Mapping[str, Any],
) -> frozenset[tuple[str, str]]:
    if not isinstance(inventory, Mapping):
        raise TypeError("A4b legacy identity inventory must be a mapping")
    if set(inventory) != set(LEGACY_IDENTITY_SCOPES):
        raise ValueError("A4b legacy identity scopes drifted")
    expected_per_task = {
        "a4_v2_formal_train": 24,
        "a4_v2_formal_development": 12,
        "a4_v2_attempted_test": 12,
    }
    identities: list[tuple[str, str]] = []
    raw_by_task: dict[str, set[int]] = {task: set() for task in TASKS}
    for scope in LEGACY_IDENTITY_SCOPES:
        rows = inventory[scope]
        if not isinstance(rows, list):
            raise TypeError("A4b legacy identity rows must be lists")
        scope_counts = {task: 0 for task in TASKS}
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError("A4b legacy identity row must be a mapping")
            if set(row) != {"task_name", "raw_task_index", "digest"}:
                raise ValueError("A4b legacy identity row keys drifted")
            digest = row.get("digest")
            task_name = row.get("task_name")
            raw_task_index = row.get("raw_task_index")
            if not isinstance(digest, str) or len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("A4b legacy reset digest is invalid")
            if task_name not in TASKS:
                raise ValueError("A4b legacy reset task identity is invalid")
            if (
                isinstance(raw_task_index, bool)
                or not isinstance(raw_task_index, int)
                or raw_task_index not in FORMAL_TASK_INSTANCES
            ):
                raise ValueError("A4b legacy raw task index is invalid")
            identities.append((str(task_name), digest))
            scope_counts[str(task_name)] += 1
            if raw_task_index in raw_by_task[str(task_name)]:
                raise ValueError("A4b legacy raw task identity is repeated")
            raw_by_task[str(task_name)].add(raw_task_index)
        if any(
            scope_counts[task] != expected_per_task[scope] for task in TASKS
        ):
            raise ValueError(f"A4b legacy scope coverage drifted for {scope}")
    if len(identities) != len(TASKS) * len(FORMAL_TASK_INSTANCES):
        raise ValueError("A4b legacy inventory must cover all 7x48 old identities")
    if len(set(identities)) != len(identities):
        raise ValueError("A4b legacy reset identities must be unique")
    if any(
        raw_by_task[task] != set(FORMAL_TASK_INSTANCES) for task in TASKS
    ):
        raise ValueError("A4b legacy raw identities must cover 0..47 per task")
    return frozenset(identities)


__all__ = [
    "A4bDrawerRetrieveEnvV1",
    "A4bEpisodeRuntime",
    "A4bTowelWipeEnvV1",
    "LEGACY_IDENTITY_SCOPES",
    "ResetPhysicalIdentity",
    "capture_reset_physical_identity",
    "collect_legacy_a4_reset_digests",
    "legacy_digest_set",
    "make_a4b_env",
    "make_a4b_episode_runtime",
    "reset_physical_state_goal_digest",
]
