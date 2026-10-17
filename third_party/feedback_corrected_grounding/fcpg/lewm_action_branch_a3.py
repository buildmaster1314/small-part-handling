"""Absolute, anonymous action branches and intervention support for A3.

The A3 bank is intentionally semantic before its final slot permutation.  Its
deployment surface contains only the eight anonymous candidate slots.  The
support gate fits whitening moments independently for every task using train
rows, then scores aligned development rows at the frozen late horizons.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np

from fcpg.lewm_action_branch_collection import ActionBranchBank
from fcpg.lewm_action_branch_metrics import (
    TrainMoments,
    whiten_latents,
)


PROTOCOL = "fcpg_action_branch_physical_support_a3_v1"
BRANCH_COUNT = 8
HORIZON = 8
FRAME_SKIP = 5
ACTION_DIM = 4
MOTION_DIM = 3
RAW_STEPS = HORIZON * FRAME_SKIP
ZERO_MOTION_TRANSVERSE_NORM = 0.15
ANONYMOUS_BRANCH_NAMES = tuple(
    f"candidate_{index}" for index in range(BRANCH_COUNT)
)
_REQUIRED_IDENTITY_KEYS = ("split", "task_index", "source_episode_id")
_OPTIONAL_IDENTITY_KEYS = (
    "task_instance_index",
    "hazard_id",
    "environment_seed",
)


def _strict_seed(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be numeric") from exc
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _action_bound(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{label} must be numeric")
    if array.ndim == 0:
        array = np.full(ACTION_DIM, float(array), dtype=np.float64)
    elif array.shape == (ACTION_DIM,):
        array = np.asarray(array, dtype=np.float64)
    else:
        raise ValueError(f"{label} must be scalar or have shape [4]")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    return array


def _validated_center(
    center_action: np.ndarray,
    *,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    center = np.asarray(center_action)
    if center.dtype == np.bool_ or not np.issubdtype(center.dtype, np.number):
        raise TypeError("center_action must be numeric")
    if center.shape != (RAW_STEPS, ACTION_DIM):
        raise ValueError("center_action must have exact shape [40,4] for H=8")
    if not np.isfinite(center).all():
        raise ValueError("center_action must contain only finite values")
    if np.any(low >= high):
        raise ValueError("action_low must be strictly below action_high")
    if np.any(low[:MOTION_DIM] >= 0.0) or np.any(high[:MOTION_DIM] <= 0.0):
        raise ValueError("A3 motion bounds must contain zero in their interior")
    center64 = np.asarray(center, dtype=np.float64)
    if np.any(center64 < low) or np.any(center64 > high):
        raise ValueError("center_action must already be inside the action bounds")
    reverse = -center64[:, :MOTION_DIM]
    if np.any(reverse < low[:MOTION_DIM]) or np.any(
        reverse > high[:MOTION_DIM]
    ):
        raise ValueError("action bounds cannot represent absolute reverse XYZ")
    center32 = np.asarray(center64, dtype=np.float32)
    if not np.isfinite(center32).all():
        raise ValueError("center_action is not representable as finite float32")
    return center32


def _transverse_frame(
    direction: np.ndarray,
    *,
    fallback_rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return two deterministic orthonormal vectors transverse to direction."""

    norm = float(np.linalg.norm(direction))
    if norm > np.finfo(np.float64).tiny:
        normal = np.asarray(direction, dtype=np.float64) / norm
    else:
        normal = fallback_rng.normal(size=MOTION_DIM)
        while float(np.linalg.norm(normal)) <= np.finfo(np.float64).tiny:
            normal = fallback_rng.normal(size=MOTION_DIM)
        normal = normal / np.linalg.norm(normal)

    reference = np.zeros(MOTION_DIM, dtype=np.float64)
    reference[int(np.argmin(np.abs(normal)))] = 1.0
    first = np.cross(normal, reference)
    first = first / np.linalg.norm(first)
    second = np.cross(normal, first)
    second = second / np.linalg.norm(second)
    return first, second


def _scale_signed_trajectory_to_bounds(
    trajectory: np.ndarray,
    *,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """Scale each XYZ step uniformly so both signs fit the action box."""

    values = np.asarray(trajectory, dtype=np.float64)
    if values.shape != (RAW_STEPS, MOTION_DIM):
        raise ValueError("transverse trajectory must have shape [40,3]")
    symmetric_limit = np.minimum(high[:MOTION_DIM], -low[:MOTION_DIM])
    ratios = np.full_like(values, np.inf)
    nonzero = np.abs(values) > np.finfo(np.float64).eps
    np.divide(
        symmetric_limit[None, :],
        np.abs(values),
        out=ratios,
        where=nonzero,
    )
    scale = np.minimum(1.0, np.min(ratios, axis=1))
    result = values * scale[:, None]
    if not np.isfinite(result).all():
        raise ValueError("direction-preserving action scaling produced non-finite XYZ")
    return result


def build_diverse_action_branch_bank(
    center_action: np.ndarray,
    *,
    direction_seed: int,
    permutation_seed: int,
    action_low: Any = -1.0,
    action_high: Any = 1.0,
    motion_delta_rms: float | None = None,
    gripper_counterfactuals: bool = False,
) -> ActionBranchBank:
    """Build the fixed A3 semantic bank and return anonymous randomized slots.

    Before permutation the XYZ trajectories are expert, absolute reverse,
    hold, half speed, and both signs of two orthogonal transverse trajectories.
    ``direction_seed`` is consulted only for a zero-motion step, where no
    expert direction exists from which to construct a transverse frame.
    """

    if motion_delta_rms is not None:
        raise ValueError("A3 uses absolute branches; motion_delta_rms must be None")
    fallback_seed = _strict_seed(direction_seed, label="direction_seed")
    slot_seed = _strict_seed(permutation_seed, label="permutation_seed")
    low = _action_bound(action_low, label="action_low")
    high = _action_bound(action_high, label="action_high")
    center = _validated_center(center_action, low=low, high=high)
    expert_xyz = np.asarray(center[:, :MOTION_DIM], dtype=np.float64)

    fallback_rng = np.random.default_rng(fallback_seed)
    transverse_first = np.empty_like(expert_xyz)
    transverse_second = np.empty_like(expert_xyz)
    for step, direction in enumerate(expert_xyz):
        first, second = _transverse_frame(
            direction,
            fallback_rng=fallback_rng,
        )
        expert_norm = float(np.linalg.norm(direction))
        transverse_norm = (
            expert_norm
            if expert_norm > np.finfo(np.float64).tiny
            else ZERO_MOTION_TRANSVERSE_NORM
        )
        transverse_first[step] = first * transverse_norm
        transverse_second[step] = second * transverse_norm
    transverse_first = _scale_signed_trajectory_to_bounds(
        transverse_first,
        low=low,
        high=high,
    )
    transverse_second = _scale_signed_trajectory_to_bounds(
        transverse_second,
        low=low,
        high=high,
    )

    if not isinstance(gripper_counterfactuals, (bool, np.bool_)):
        raise TypeError("gripper_counterfactuals must be boolean")
    semantic_xyz = (
        expert_xyz,
        -expert_xyz,
        np.zeros_like(expert_xyz),
        0.5 * expert_xyz,
        transverse_first,
        -transverse_first,
        expert_xyz if gripper_counterfactuals else transverse_second,
        expert_xyz if gripper_counterfactuals else -transverse_second,
    )
    semantic_bank = np.broadcast_to(
        center[None, :, :],
        (BRANCH_COUNT, RAW_STEPS, ACTION_DIM),
    ).copy()
    semantic_bank[:, :, :MOTION_DIM] = np.stack(semantic_xyz).astype(
        np.float32,
        copy=False,
    )
    semantic_bank[:, :, MOTION_DIM] = center[None, :, MOTION_DIM]
    if gripper_counterfactuals:
        semantic_bank[6, :, MOTION_DIM] = low[MOTION_DIM]
        semantic_bank[7, :, MOTION_DIM] = high[MOTION_DIM]

    permutation = np.random.default_rng(slot_seed).permutation(BRANCH_COUNT)
    actions = np.ascontiguousarray(semantic_bank[permutation], dtype=np.float32)
    if actions.shape != (BRANCH_COUNT, RAW_STEPS, ACTION_DIM):
        raise RuntimeError("A3 branch construction produced an invalid shape")
    if not np.isfinite(actions).all():
        raise RuntimeError("A3 branch construction produced non-finite actions")
    if np.any(actions < low) or np.any(actions > high):
        raise RuntimeError("A3 branch construction produced out-of-bound actions")
    semantic_gripper = np.broadcast_to(
        center[None, :, MOTION_DIM],
        (BRANCH_COUNT, RAW_STEPS),
    ).copy()
    if gripper_counterfactuals:
        semantic_gripper[6] = low[MOTION_DIM]
        semantic_gripper[7] = high[MOTION_DIM]
    expected_gripper = semantic_gripper[permutation]
    if not np.array_equal(actions[:, :, MOTION_DIM], expected_gripper):
        raise RuntimeError("A3 branches produced unexpected gripper commands")
    return ActionBranchBank(
        names=ANONYMOUS_BRANCH_NAMES,
        center_action=center,
        actions=actions,
    )


def _required_vector(
    arrays: Mapping[str, np.ndarray],
    key: str,
    *,
    count: int,
) -> np.ndarray:
    if key not in arrays:
        raise KeyError(f"arrays are missing required key {key!r}")
    value = np.asarray(arrays[key])
    if value.shape != (count,):
        raise ValueError(f"{key} must have shape [N]")
    return value


def _validated_target(latent_arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    if not isinstance(latent_arrays, Mapping):
        raise TypeError("latent_arrays must be a mapping")
    if "target_embedding" not in latent_arrays:
        raise KeyError("latent_arrays are missing required key 'target_embedding'")
    target = np.asarray(latent_arrays["target_embedding"])
    if target.dtype == np.bool_ or not np.issubdtype(target.dtype, np.number):
        raise TypeError("target_embedding must be numeric")
    if target.ndim != 5 or target.shape[1:3] != (
        BRANCH_COUNT,
        HORIZON,
    ):
        raise ValueError("target_embedding must have shape [N,8,8,K,D]")
    if target.shape[0] == 0 or target.shape[3] == 0 or target.shape[4] == 0:
        raise ValueError("target_embedding axes must be non-empty")
    if not np.isfinite(target).all():
        raise ValueError("target_embedding must contain only finite values")
    return target


def _validate_aligned_physical_arrays(
    latent_arrays: Mapping[str, np.ndarray],
    physical_arrays: Mapping[str, np.ndarray],
    *,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(physical_arrays, Mapping):
        raise TypeError("physical_arrays must be a mapping")
    for key in _REQUIRED_IDENTITY_KEYS:
        latent_value = _required_vector(latent_arrays, key, count=count)
        physical_value = _required_vector(physical_arrays, key, count=count)
        if not np.array_equal(latent_value, physical_value):
            raise ValueError(f"physical {key} is not aligned with latent rows")
    for key in _OPTIONAL_IDENTITY_KEYS:
        in_latent = key in latent_arrays
        in_physical = key in physical_arrays
        if in_latent != in_physical:
            raise ValueError(f"physical and latent identity key {key!r} must align")
        if in_latent:
            latent_value = _required_vector(latent_arrays, key, count=count)
            physical_value = _required_vector(physical_arrays, key, count=count)
            if not np.array_equal(latent_value, physical_value):
                raise ValueError(f"physical {key} is not aligned with latent rows")

    result: list[np.ndarray] = []
    for key in ("tcp", "object_position"):
        if key not in physical_arrays:
            raise KeyError(f"physical_arrays are missing required key {key!r}")
        value = np.asarray(physical_arrays[key])
        if value.dtype == np.bool_ or not np.issubdtype(value.dtype, np.number):
            raise TypeError(f"physical {key} must be numeric")
        if value.shape != (count, BRANCH_COUNT, HORIZON, MOTION_DIM):
            raise ValueError(f"physical {key} must have shape [N,8,8,3]")
        if not np.isfinite(value).all():
            raise ValueError(f"physical {key} must contain only finite values")
        result.append(np.asarray(value, dtype=np.float64))
    return result[0], result[1]


def _fit_task_train_moments(
    target: np.ndarray,
    task_index: np.ndarray,
    split: np.ndarray,
    episode: np.ndarray,
    horizon_indices: np.ndarray,
    *,
    task_count: int,
) -> dict[int, TrainMoments]:
    """Fit H6-H8 moments with physical train episodes weighted equally."""

    moments: dict[int, TrainMoments] = {}
    for index in range(task_count):
        selected = (task_index == index) & (split == "train")
        if not np.any(selected):
            raise ValueError(f"task_index {index} has no train rows")
        selected_target = np.asarray(
            target[selected][:, :, horizon_indices], dtype=np.float64
        )
        selected_episode = episode[selected]
        episode_means: list[np.ndarray] = []
        episode_second_moments: list[np.ndarray] = []
        for identity in np.unique(selected_episode):
            values = selected_target[selected_episode == identity]
            episode_means.append(np.mean(values, axis=(0, 1, 2), dtype=np.float64))
            episode_second_moments.append(
                np.mean(np.square(values), axis=(0, 1, 2), dtype=np.float64)
            )
        mean = np.mean(episode_means, axis=0, dtype=np.float64)
        second = np.mean(episode_second_moments, axis=0, dtype=np.float64)
        raw_scale = np.sqrt(np.maximum(second - np.square(mean), 0.0))
        floored = raw_scale <= 1e-8
        moments[index] = TrainMoments(
            mean=mean,
            scale=np.where(floored, 1.0, raw_scale),
            count=len(episode_means),
            epsilon=1e-8,
            floored_dimensions=int(np.count_nonzero(floored)),
        )
    return moments


def _late_pairwise_spread(
    position: np.ndarray,
    horizon_indices: np.ndarray,
) -> np.ndarray:
    selected = position[:, :, horizon_indices, :]
    delta = selected[:, :, None, :, :] - selected[:, None, :, :, :]
    distance = np.linalg.norm(delta, axis=-1)
    return np.max(distance, axis=(1, 2, 3))


def _episode_macro(
    informative: np.ndarray,
    episode: np.ndarray,
) -> tuple[float, np.ndarray]:
    identities = np.unique(episode)
    fractions = np.asarray(
        [np.mean(informative[episode == identity]) for identity in identities],
        dtype=np.float64,
    )
    return float(np.mean(fractions)), fractions


def summarize_a3_support(
    latent_arrays: Mapping[str, np.ndarray],
    physical_arrays: Mapping[str, np.ndarray],
    *,
    task_names: Sequence[str],
    primary_horizons: Sequence[int] = (6, 7, 8),
    minimum_latent_variance: float = 0.01,
    minimum_tcp_spread_m: float = 0.01,
    minimum_object_spread_m: float = 0.005,
    minimum_episode_macro_fraction: float = 0.8,
    expected_train_episodes: int = 24,
    expected_development_episodes: int = 12,
) -> dict[str, Any]:
    """Summarize the conjunctive A3 physical and latent support gate."""

    names = tuple(str(name) for name in task_names)
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("task_names must be distinct and non-empty")
    horizons = tuple(int(value) for value in primary_horizons)
    if not horizons or len(set(horizons)) != len(horizons):
        raise ValueError("primary_horizons must be distinct and non-empty")
    if any(value < 1 or value > HORIZON for value in horizons):
        raise ValueError("primary_horizons must be in the range 1..8")
    horizon_indices = np.asarray([value - 1 for value in horizons], dtype=np.int64)

    latent_threshold = _finite_number(
        minimum_latent_variance,
        label="minimum_latent_variance",
    )
    tcp_threshold = _finite_number(
        minimum_tcp_spread_m,
        label="minimum_tcp_spread_m",
    )
    object_threshold = _finite_number(
        minimum_object_spread_m,
        label="minimum_object_spread_m",
    )
    macro_threshold = _finite_number(
        minimum_episode_macro_fraction,
        label="minimum_episode_macro_fraction",
    )
    if min(latent_threshold, tcp_threshold, object_threshold) < 0.0:
        raise ValueError("A3 support thresholds must be non-negative")
    if not 0.0 <= macro_threshold <= 1.0:
        raise ValueError("minimum_episode_macro_fraction must be in [0,1]")
    train_episode_target = _strict_seed(
        expected_train_episodes,
        label="expected_train_episodes",
    )
    development_episode_target = _strict_seed(
        expected_development_episodes,
        label="expected_development_episodes",
    )
    if train_episode_target == 0 or development_episode_target == 0:
        raise ValueError("expected episode counts must be positive")

    target = _validated_target(latent_arrays)
    count = target.shape[0]
    split = _required_vector(latent_arrays, "split", count=count).astype(str)
    if np.any(split == "test"):
        raise ValueError("A3 support refuses arrays containing test rows")
    if np.any(~np.isin(split, ("train", "development"))):
        raise ValueError("A3 support requires only train and development rows")
    task_index_raw = _required_vector(latent_arrays, "task_index", count=count)
    if task_index_raw.dtype == np.bool_ or not np.issubdtype(
        task_index_raw.dtype,
        np.integer,
    ):
        raise TypeError("task_index must be an integer array")
    task_index = np.asarray(task_index_raw, dtype=np.int64)
    expected_indices = set(range(len(names)))
    if set(task_index.tolist()) != expected_indices:
        raise ValueError(
            f"task_index must cover {sorted(expected_indices)} exactly"
        )
    episode = _required_vector(
        latent_arrays,
        "source_episode_id",
        count=count,
    ).astype(str)
    if np.any(episode == ""):
        raise ValueError("source_episode_id values must be non-empty")
    tcp, object_position = _validate_aligned_physical_arrays(
        latent_arrays,
        physical_arrays,
        count=count,
    )

    episode_split: dict[tuple[int, str], str] = {}
    for task, identity, label in zip(task_index, episode, split, strict=True):
        key = (int(task), str(identity))
        previous = episode_split.setdefault(key, str(label))
        if previous != label:
            raise ValueError("a source physical episode crosses train/development")

    moments = _fit_task_train_moments(
        target,
        task_index,
        split,
        episode,
        horizon_indices,
        task_count=len(names),
    )
    development = split == "development"
    tcp_spread = _late_pairwise_spread(tcp[development], horizon_indices)
    object_spread = _late_pairwise_spread(
        object_position[development],
        horizon_indices,
    )
    physical_informative = (tcp_spread >= tcp_threshold) | (
        object_spread >= object_threshold
    )

    development_indices = np.flatnonzero(development)
    latent_variance = np.empty(development_indices.size, dtype=np.float64)
    for index in range(len(names)):
        selected = task_index[development_indices] == index
        task_target = target[development_indices[selected]]
        whitened = whiten_latents(
            task_target,
            moments[index],
            label=f"{names[index]} development target",
        )
        centered = whitened - np.mean(
            whitened,
            axis=1,
            keepdims=True,
            dtype=np.float64,
        )
        late = centered[:, :, horizon_indices, :, :]
        latent_variance[selected] = np.mean(
            np.square(late),
            axis=(1, 2, 3, 4),
            dtype=np.float64,
        )
    latent_informative = latent_variance >= latent_threshold

    development_task = task_index[development]
    development_episode = episode[development]
    per_task: dict[str, Any] = {}
    for index, name in enumerate(names):
        train_episode_count = int(
            np.unique(episode[(task_index == index) & (split == "train")]).size
        )
        development_episode_count = int(
            np.unique(
                episode[(task_index == index) & (split == "development")]
            ).size
        )
        if (
            train_episode_count != train_episode_target
            or development_episode_count != development_episode_target
        ):
            raise ValueError(
                f"A3 {name} requires exactly {train_episode_target} train and "
                f"{development_episode_target} development episodes, got "
                f"{train_episode_count}/{development_episode_count}"
            )
        selected = development_task == index
        task_episode = development_episode[selected]
        task_physical = physical_informative[selected]
        task_latent = latent_informative[selected]
        task_joint = task_physical & task_latent
        physical_macro, physical_episode_fractions = _episode_macro(
            task_physical,
            task_episode,
        )
        latent_macro, latent_episode_fractions = _episode_macro(
            task_latent,
            task_episode,
        )
        joint_macro, joint_episode_fractions = _episode_macro(
            task_joint,
            task_episode,
        )
        physical_passed = physical_macro >= macro_threshold
        latent_passed = latent_macro >= macro_threshold
        joint_passed = joint_macro >= macro_threshold
        median_tcp_spread = float(np.median(tcp_spread[selected]))
        median_object_spread = float(np.median(object_spread[selected]))
        median_latent_variance = float(np.median(latent_variance[selected]))
        per_task[name] = {
            "bundle_count": int(np.count_nonzero(selected)),
            "train_episode_count": train_episode_count,
            "development_episode_count": development_episode_count,
            "physical_episode_macro_informative_fraction": physical_macro,
            "latent_episode_macro_informative_fraction": latent_macro,
            "joint_episode_macro_informative_fraction": joint_macro,
            "median_tcp_max_pairwise_m": median_tcp_spread,
            "median_object_max_pairwise_m": median_object_spread,
            "median_latent_branch_variance": median_latent_variance,
            "physical": {
                "informative_bundle_count": int(np.count_nonzero(task_physical)),
                "bundle_fraction": float(np.mean(task_physical)),
                "episode_macro_informative_fraction": physical_macro,
                "minimum_episode_fraction": float(
                    np.min(physical_episode_fractions)
                ),
                "median_tcp_spread_m": median_tcp_spread,
                "median_object_spread_m": median_object_spread,
                "passed": bool(physical_passed),
            },
            "latent": {
                "informative_bundle_count": int(np.count_nonzero(task_latent)),
                "bundle_fraction": float(np.mean(task_latent)),
                "episode_macro_informative_fraction": latent_macro,
                "minimum_episode_fraction": float(np.min(latent_episode_fractions)),
                "median_branch_variance": median_latent_variance,
                "train_moment_count": int(moments[index].count),
                "train_moment_floored_dimensions": int(
                    moments[index].floored_dimensions
                ),
                "passed": bool(latent_passed),
            },
            "joint": {
                "informative_bundle_count": int(np.count_nonzero(task_joint)),
                "bundle_fraction": float(np.mean(task_joint)),
                "episode_macro_informative_fraction": joint_macro,
                "minimum_episode_fraction": float(np.min(joint_episode_fractions)),
                "passed": bool(joint_passed),
            },
            "passed": bool(joint_passed),
        }

    physical_passed = all(row["physical"]["passed"] for row in per_task.values())
    latent_passed = all(row["latent"]["passed"] for row in per_task.values())
    joint_passed = all(row["joint"]["passed"] for row in per_task.values())
    passed = joint_passed
    return {
        "protocol": PROTOCOL,
        "stage": "collection_only_physical_and_latent_support",
        "decision": "Go" if passed else "No-Go",
        "model_training_started": False,
        "input_contains_test_rows": False,
        "primary_horizons": list(horizons),
        "minimum_latent_variance": latent_threshold,
        "minimum_tcp_spread_m": tcp_threshold,
        "minimum_object_spread_m": object_threshold,
        "minimum_episode_macro_fraction": macro_threshold,
        "expected_train_episodes": train_episode_target,
        "expected_development_episodes": development_episode_target,
        "physical_support_passed": bool(physical_passed),
        "latent_support_passed": bool(latent_passed),
        "joint_support_passed": bool(joint_passed),
        "per_task": per_task,
    }


__all__ = [
    "build_diverse_action_branch_bank",
    "summarize_a3_support",
]
