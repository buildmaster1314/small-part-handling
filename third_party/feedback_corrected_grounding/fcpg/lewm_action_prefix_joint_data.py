"""Raw A4 replay storage for joint seven-task visual-dynamics training."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fcpg.lewm_action_branch_a4 import BRANCH_COUNT, HORIZON, TASKS


SCHEMA_VERSION = 1
RAW_ACTION_DIM = 4
MODEL_ACTION_DIM = 20
NUM_KEYPOINTS = 3


def accepted_ledger_rows(
    cache: Mapping[str, np.ndarray],
    ledger: Mapping[str, np.ndarray],
) -> dict[int, int]:
    """Map each cache bundle index to its unique accepted ledger row."""

    bundle_count = int(np.asarray(cache["proposal_id"]).shape[0])
    accepted = np.asarray(ledger["accepted_bundle_index"], dtype=np.int64)
    mapping: dict[int, int] = {}
    for ledger_index, bundle_index in enumerate(accepted.tolist()):
        if bundle_index < 0:
            continue
        if bundle_index >= bundle_count:
            raise ValueError("accepted bundle index is outside the state cache")
        if bundle_index in mapping:
            raise ValueError("accepted bundle index appears more than once")
        if str(ledger["proposal_id"][ledger_index]) != str(
            cache["proposal_id"][bundle_index]
        ):
            raise ValueError("proposal identity differs between ledger and cache")
        mapping[bundle_index] = ledger_index
    if len(mapping) != bundle_count:
        raise ValueError("ledger does not cover every accepted state-cache bundle")
    return mapping


def _round_robin_episode_indices(
    indices: Sequence[int],
    cache: Mapping[str, np.ndarray],
    ledger: Mapping[str, np.ndarray],
    ledger_rows: Mapping[int, int],
    count: int,
) -> list[int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        groups[str(cache["source_episode_id"][index])].append(int(index))
    queues = []
    for episode in sorted(groups):
        ordered = sorted(
            groups[episode],
            key=lambda index: int(
                ledger["proposal_raw_step"][ledger_rows[index]]
            ),
        )
        queues.append(deque(ordered))
    selected: list[int] = []
    while queues and len(selected) < count:
        next_queues = []
        for queue in queues:
            if queue and len(selected) < count:
                selected.append(queue.popleft())
            if queue:
                next_queues.append(queue)
        queues = next_queues
    if len(selected) != count:
        raise ValueError(
            f"requested {count} balanced bundles but only found {len(selected)}"
        )
    return selected


def select_balanced_bundle_indices(
    cache: Mapping[str, np.ndarray],
    ledger: Mapping[str, np.ndarray],
    *,
    train_per_task: int,
    development_per_task: int,
) -> np.ndarray:
    """Select task-balanced bundles while spreading them across source episodes."""

    if train_per_task <= 0 or development_per_task <= 0:
        raise ValueError("per-task bundle counts must be positive")
    ledger_rows = accepted_ledger_rows(cache, ledger)
    task_names = np.asarray(cache["task_name"]).astype(str)
    splits = np.asarray(cache["split"]).astype(str)
    selected: list[int] = []
    for task in TASKS:
        for split, count in (
            ("train", int(train_per_task)),
            ("development", int(development_per_task)),
        ):
            candidates = np.flatnonzero(
                (task_names == task) & (splits == split)
            ).tolist()
            selected.extend(
                _round_robin_episode_indices(
                    candidates,
                    cache,
                    ledger,
                    ledger_rows,
                    count,
                )
            )
    result = np.asarray(selected, dtype=np.int64)
    if np.unique(result).size != result.size:
        raise RuntimeError("balanced bundle selection contains duplicates")
    return result


def _phase_schedule_lookup(
    phase_schedule: Mapping[str, Any],
) -> dict[tuple[str, int, int], str]:
    episodes = phase_schedule.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("phase schedule must contain episode rows")
    lookup: dict[tuple[str, int, int], str] = {}
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise TypeError("phase schedule episode must be a mapping")
        steps = episode.get("proposal_steps")
        coverage = episode.get("coverage")
        phases = (
            coverage.get("selected_phase_names")
            if isinstance(coverage, Mapping)
            else None
        )
        if not isinstance(steps, list) or not isinstance(phases, list):
            raise ValueError("phase schedule episode lacks proposal phases")
        if len(steps) != len(phases):
            raise ValueError("phase schedule steps and phases differ in length")
        task = str(episode["task"])
        task_instance = int(episode["task_instance"])
        for step, phase in zip(steps, phases):
            key = (task, task_instance, int(step))
            if key in lookup:
                raise ValueError("phase schedule contains a duplicate proposal")
            value = str(phase)
            if not value:
                raise ValueError("phase schedule contains an empty phase")
            lookup[key] = value
    return lookup


def phase_labels_for_bundle_indices(
    cache: Mapping[str, np.ndarray],
    ledger: Mapping[str, np.ndarray],
    selected_indices: Sequence[int],
    phase_schedule: Mapping[str, Any],
) -> np.ndarray:
    """Return the planned task phase for each selected physical bundle."""

    ledger_rows = accepted_ledger_rows(cache, ledger)
    lookup = _phase_schedule_lookup(phase_schedule)
    labels: list[str] = []
    for cache_index in selected_indices:
        index = int(cache_index)
        ledger_index = ledger_rows[index]
        key = (
            str(cache["task_name"][index]),
            int(cache["task_instance_index"][index]),
            int(ledger["proposal_raw_step"][ledger_index]),
        )
        try:
            labels.append(lookup[key])
        except KeyError as exc:
            raise ValueError(
                "accepted bundle is absent from the phase schedule"
            ) from exc
    return np.asarray(labels)


def _phase_spread_episode_indices(
    groups: Mapping[str, Sequence[int]],
    ledger: Mapping[str, np.ndarray],
    ledger_rows: Mapping[int, int],
    count: int,
) -> list[int]:
    episodes = sorted(groups)
    available = sum(len(groups[episode]) for episode in episodes)
    if count > available:
        raise ValueError(
            f"requested {count} phase-balanced bundles but only found {available}"
        )
    pass_count = (count + len(episodes) - 1) // len(episodes)
    selected: list[int] = []
    selected_set: set[int] = set()
    for pass_index in range(pass_count):
        remaining = count - len(selected)
        take = min(len(episodes), remaining)
        episode_ranks = np.floor(
            (np.arange(take, dtype=np.float64) + 0.5)
            * len(episodes)
            / take
        ).astype(np.int64)
        for episode_rank in episode_ranks.tolist():
            episode = episodes[int(episode_rank)]
            ordered = sorted(
                (int(index) for index in groups[episode]),
                key=lambda index: int(
                    ledger["proposal_raw_step"][ledger_rows[index]]
                ),
            )
            candidates = [index for index in ordered if index not in selected_set]
            if not candidates:
                continue
            fraction = (
                (float(episode_rank) + 0.5) / len(episodes)
                + float(pass_index) / pass_count
            ) % 1.0
            target = fraction * max(len(ordered) - 1, 0)
            chosen = min(
                candidates,
                key=lambda index: (
                    abs(float(ordered.index(index)) - target),
                    int(ledger["proposal_raw_step"][ledger_rows[index]]),
                ),
            )
            selected.append(chosen)
            selected_set.add(chosen)
    if len(selected) != count:
        raise RuntimeError("phase-balanced selection did not fill its budget")
    return selected


def select_phase_balanced_bundle_indices(
    cache: Mapping[str, np.ndarray],
    ledger: Mapping[str, np.ndarray],
    phase_schedule: Mapping[str, Any],
    *,
    train_per_task: int,
    development_per_task: int,
) -> np.ndarray:
    """Select episode- and trajectory-phase-balanced A4c bundles."""

    if train_per_task <= 0 or development_per_task <= 0:
        raise ValueError("per-task bundle counts must be positive")
    ledger_rows = accepted_ledger_rows(cache, ledger)
    lookup = _phase_schedule_lookup(phase_schedule)
    task_names = np.asarray(cache["task_name"]).astype(str)
    task_instances = np.asarray(cache["task_instance_index"], dtype=np.int64)
    source_episodes = np.asarray(cache["source_episode_id"]).astype(str)
    splits = np.asarray(cache["split"]).astype(str)
    selected: list[int] = []
    for task in TASKS:
        for split, count in (
            ("train", int(train_per_task)),
            ("development", int(development_per_task)),
        ):
            candidates = np.flatnonzero(
                (task_names == task) & (splits == split)
            ).tolist()
            groups: dict[str, list[int]] = defaultdict(list)
            for index in candidates:
                key = (
                    task,
                    int(task_instances[index]),
                    int(ledger["proposal_raw_step"][ledger_rows[index]]),
                )
                if key not in lookup:
                    raise ValueError(
                        "accepted A4c bundle is absent from the phase schedule"
                    )
                groups[str(source_episodes[index])].append(int(index))
            if not groups:
                raise ValueError(f"no phase-balanced candidates for {task} {split}")
            selected.extend(
                _phase_spread_episode_indices(
                    groups,
                    ledger,
                    ledger_rows,
                    count,
                )
            )
    result = np.asarray(selected, dtype=np.int64)
    if np.unique(result).size != result.size:
        raise RuntimeError("phase-balanced bundle selection contains duplicates")
    phase_labels_for_bundle_indices(
        cache, ledger, result.tolist(), phase_schedule
    )
    return result


def replay_index_payload(
    cache: Mapping[str, np.ndarray],
    ledger: Mapping[str, np.ndarray],
    selected_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build the small immutable index stored beside raw memmaps."""

    ledger_rows = accepted_ledger_rows(cache, ledger)
    selected = np.asarray(selected_indices, dtype=np.int64)
    rows = np.asarray([ledger_rows[int(index)] for index in selected], dtype=np.int64)
    payload = {
        "cache_index": selected,
        "proposal_id": np.asarray(cache["proposal_id"])[selected],
        "task_index": np.asarray(cache["task_class_index"], dtype=np.int64)[selected],
        "task_name": np.asarray(cache["task_name"])[selected],
        "task_instance": np.asarray(cache["task_instance_index"], dtype=np.int64)[selected],
        "environment_seed": np.asarray(ledger["environment_seed"], dtype=np.int64)[rows],
        "proposal_raw_step": np.asarray(ledger["proposal_raw_step"], dtype=np.int64)[rows],
        "source_episode_id": np.asarray(cache["source_episode_id"])[selected],
        "split": np.asarray(cache["split"])[selected],
        "future_action": np.asarray(cache["future_action"], dtype=np.float32)[selected],
    }
    validate_replay_index(payload)
    return payload


def validate_replay_index(payload: Mapping[str, np.ndarray]) -> int:
    required = {
        "cache_index",
        "proposal_id",
        "task_index",
        "task_name",
        "task_instance",
        "environment_seed",
        "proposal_raw_step",
        "source_episode_id",
        "split",
        "future_action",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"raw replay index is missing fields: {missing}")
    count = int(np.asarray(payload["cache_index"]).shape[0])
    if count <= 0:
        raise ValueError("raw replay index must contain at least one bundle")
    vector_fields = required - {"future_action"}
    for name in vector_fields:
        if np.asarray(payload[name]).shape != (count,):
            raise ValueError(f"raw replay index field {name} must have shape {(count,)}")
    actions = np.asarray(payload["future_action"])
    if actions.shape != (count, BRANCH_COUNT, HORIZON, MODEL_ACTION_DIM):
        raise ValueError("raw replay future_action must have shape [N,8,8,20]")
    if actions.dtype.kind != "f" or not np.isfinite(actions).all():
        raise ValueError("raw replay future actions must be finite floating point")
    if not set(np.asarray(payload["split"]).astype(str)) <= {
        "train",
        "development",
    }:
        raise ValueError("raw replay index cannot contain reserved-test rows")
    task_names = np.asarray(payload["task_name"]).astype(str)
    task_indices = np.asarray(payload["task_index"], dtype=np.int64)
    for task_index, task in enumerate(TASKS):
        mask = task_names == task
        if bool(mask.any()) and not bool(np.all(task_indices[mask] == task_index)):
            raise ValueError("raw replay task names and indices disagree")
    if not set(task_names.tolist()) <= set(TASKS):
        raise ValueError("raw replay index contains an unknown task")
    optional_camera = {"camera_index", "camera_name", "physical_bundle_index"}
    present_camera = optional_camera.intersection(payload)
    if present_camera and present_camera != optional_camera:
        raise ValueError("raw replay camera identity fields must be stored together")
    if present_camera:
        for name in optional_camera:
            if np.asarray(payload[name]).shape != (count,):
                raise ValueError(f"raw replay field {name} must have shape {(count,)}")
        camera_index = np.asarray(payload["camera_index"], dtype=np.int64)
        camera_name = np.asarray(payload["camera_name"]).astype(str)
        if int(camera_index.min()) < 0 or not bool(np.all(camera_name != "")):
            raise ValueError("raw replay camera identity contains invalid values")
        by_index: dict[int, str] = {}
        for index_value, name_value in zip(camera_index.tolist(), camera_name.tolist()):
            previous = by_index.setdefault(int(index_value), str(name_value))
            if previous != str(name_value):
                raise ValueError("camera index maps to more than one camera name")
    optional_orbit = {
        "camera_base_name",
        "camera_clock_hour",
        "camera_elevation_degrees",
        "camera_radius_m",
    }
    present_orbit = optional_orbit.intersection(payload)
    if present_orbit and present_orbit != optional_orbit:
        raise ValueError("raw replay orbit camera fields must be stored together")
    if present_orbit:
        if not present_camera:
            raise ValueError("orbit camera metadata requires camera identity fields")
        for name in optional_orbit:
            if np.asarray(payload[name]).shape != (count,):
                raise ValueError(f"raw replay field {name} must have shape {(count,)}")
        base_name = np.asarray(payload["camera_base_name"]).astype(str)
        clock_hour = np.asarray(payload["camera_clock_hour"], dtype=np.int64)
        elevation = np.asarray(payload["camera_elevation_degrees"], dtype=np.float64)
        radius = np.asarray(payload["camera_radius_m"], dtype=np.float64)
        if not bool(np.all(base_name != "")):
            raise ValueError("orbit base camera names must be non-empty")
        if int(clock_hour.min()) < 1 or int(clock_hour.max()) > 12:
            raise ValueError("orbit clock hours must be in [1,12]")
        if not np.isfinite(elevation).all() or not bool(
            np.all((elevation >= 5.0) & (elevation <= 80.0))
        ):
            raise ValueError("orbit elevations must be finite and in [5,80]")
        if not np.isfinite(radius).all() or not bool(
            np.all((radius >= 0.3) & (radius <= 5.0))
        ):
            raise ValueError("orbit radii must be finite and in [0.3,5.0]")
    if "proposal_phase_name" in payload:
        phase_name = np.asarray(payload["proposal_phase_name"]).astype(str)
        if phase_name.shape != (count,) or not bool(np.all(phase_name != "")):
            raise ValueError("proposal phase names must be non-empty per bundle")
    return count


def open_raw_replay_arrays(
    root: str | Path,
    *,
    mode: str = "r",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load the replay index and memory-mapped observation arrays."""

    directory = Path(root).expanduser().resolve()
    with np.load(directory / "index.npz", allow_pickle=False) as archive:
        index = {name: np.array(archive[name], copy=True) for name in archive.files}
    count = validate_replay_index(index)
    arrays = {
        "current_rgb": np.load(directory / "current_rgb.npy", mmap_mode=mode),
        "current_proprio": np.load(directory / "current_proprio.npy", mmap_mode=mode),
        "current_q_nominal": np.load(directory / "current_q_nominal.npy", mmap_mode=mode),
        "current_q_corrected": np.load(directory / "current_q_corrected.npy", mmap_mode=mode),
        "current_keypoints_xyz": np.load(directory / "current_keypoints_xyz.npy", mmap_mode=mode),
        "current_valid": np.load(directory / "current_valid.npy", mmap_mode=mode),
        "current_q_true": np.load(directory / "current_q_true.npy", mmap_mode=mode),
        "future_rgb": np.load(directory / "future_rgb.npy", mmap_mode=mode),
        "future_proprio": np.load(directory / "future_proprio.npy", mmap_mode=mode),
        "future_q_nominal": np.load(directory / "future_q_nominal.npy", mmap_mode=mode),
        "future_q_corrected": np.load(directory / "future_q_corrected.npy", mmap_mode=mode),
        "future_keypoints_xyz": np.load(directory / "future_keypoints_xyz.npy", mmap_mode=mode),
        "future_valid": np.load(directory / "future_valid.npy", mmap_mode=mode),
        "future_q_true": np.load(directory / "future_q_true.npy", mmap_mode=mode),
        "completed": np.load(directory / "completed.npy", mmap_mode=mode),
    }
    optional_camera_arrays = {
        "camera_intrinsic": ("camera_intrinsic.npy", (count, 3, 3)),
        "camera_extrinsic_cam_from_world": (
            "camera_extrinsic_cam_from_world.npy",
            (count, 4, 4),
        ),
    }
    existing_camera_arrays = {
        name: directory / filename
        for name, (filename, _shape) in optional_camera_arrays.items()
        if (directory / filename).exists()
    }
    if existing_camera_arrays and len(existing_camera_arrays) != len(
        optional_camera_arrays
    ):
        raise ValueError("raw replay camera matrix arrays must be stored together")
    for name, (filename, expected_shape) in optional_camera_arrays.items():
        path = directory / filename
        if path.exists():
            value = np.load(path, mmap_mode=mode)
            if value.shape != expected_shape or value.dtype.kind != "f":
                raise ValueError(f"raw replay camera matrix {name} has invalid shape")
            if not np.isfinite(value).all():
                raise ValueError(f"raw replay camera matrix {name} is non-finite")
            arrays[name] = value
    if arrays["current_rgb"].shape[0] != count:
        raise ValueError("raw replay observation count differs from index")
    if arrays["future_rgb"].shape[:3] != (count, BRANCH_COUNT, HORIZON):
        raise ValueError("raw replay future RGB axes drifted")
    if arrays["completed"].shape != (count,) or not bool(arrays["completed"].all()):
        raise RuntimeError("raw replay dataset is incomplete")
    return index, arrays


__all__ = [
    "SCHEMA_VERSION",
    "accepted_ledger_rows",
    "open_raw_replay_arrays",
    "phase_labels_for_bundle_indices",
    "replay_index_payload",
    "select_balanced_bundle_indices",
    "select_phase_balanced_bundle_indices",
    "validate_replay_index",
]
