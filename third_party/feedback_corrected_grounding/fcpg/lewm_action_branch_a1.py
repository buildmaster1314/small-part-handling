"""Core array contract and action branches for the A1 action-effect study.

A branch center is the complete future raw-action sequence from the source
scripted expert.  In particular, an A1 horizon of eight requires 40 supplied
raw actions.  This module never pads, truncates, or repeats actions to create a
center.  The eight branches identify action effects from one restored physical
state; they are not deployment or CEM candidates.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


HISTORY_STEPS = 3
CANDIDATE_COUNT = 8
AXIS_COUNT = CANDIDATE_COUNT // 2
MIN_HORIZON = 8
RAW_STEPS_PER_HORIZON = 5
RAW_ACTION_DIM = 4
MOTION_DIM = 3
ACTION_DIM = RAW_STEPS_PER_HORIZON * RAW_ACTION_DIM
KEYPOINT_COUNT = 3
EMBEDDING_DIM = 192
TARGET_MOTION_RMS = 0.15
FORMAL_TASK_INSTANCE_COUNT = 48
SMOKE_TASK_INSTANCE_COUNT = 2
TASK_INSTANCE_COUNT = FORMAL_TASK_INSTANCE_COUNT + SMOKE_TASK_INSTANCE_COUNT

RMS_ABSOLUTE_TOLERANCE = 2.0e-6
ORTHONORMAL_ABSOLUTE_TOLERANCE = 1.0e-10
MIN_PAIRWISE_MOTION_RMS = 0.90 * TARGET_MOTION_RMS
MAX_PAIRWISE_MOTION_COSINE = 0.60
BISECTION_STEPS = 96

NPZ_KEYS = (
    "history_embedding",
    "history_action",
    "center_action",
    "future_action",
    "target_embedding",
    "task_index",
    "task_instance_index",
    "source_episode_id",
    "hazard_id",
    "split",
)
VALID_SPLITS = ("train", "development", "test", "smoke")
ANONYMOUS_BRANCH_NAMES = tuple(
    f"candidate_{index}" for index in range(CANDIDATE_COUNT)
)


@dataclass(frozen=True, slots=True)
class A1ArrayAxes:
    """Validated dimensions of one aligned A1 NPZ bundle."""

    sample_count: int
    history_steps: int
    candidate_count: int
    horizon: int
    action_dim: int
    keypoint_count: int
    embedding_dim: int


@dataclass(frozen=True, slots=True)
class ActionBranchValidation:
    """Measurements proving the public branch array is executor-ready."""

    horizon: int
    realized_motion_rms: tuple[float, ...]
    minimum_pairwise_motion_rms: float
    maximum_pairwise_motion_cosine: float


def _exact_keys(arrays: Mapping[str, np.ndarray]) -> None:
    actual = set(arrays)
    expected = set(NPZ_KEYS)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"A1 NPZ keys changed: missing={missing}, extra={extra}")


def _finite_numeric(value: np.ndarray, name: str) -> None:
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite numeric data")


def _integer_vector(value: np.ndarray, name: str, sample_count: int) -> None:
    if value.shape != (sample_count,):
        raise ValueError(f"{name} must have shape [N]")
    if not np.issubdtype(value.dtype, np.integer):
        raise TypeError(f"{name} must contain integers")


def _identifier_vector(value: np.ndarray, name: str, sample_count: int) -> None:
    if value.shape != (sample_count,):
        raise ValueError(f"{name} must have shape [N]")
    if np.issubdtype(value.dtype, np.integer):
        return
    if value.dtype.kind not in {"U", "S"}:
        raise TypeError(f"{name} must contain integers or fixed-width strings")
    if any(not item.strip() for item in value.astype(str)):
        raise ValueError(f"{name} must not contain empty strings")


def _split_vector(value: np.ndarray, sample_count: int) -> np.ndarray:
    if value.shape != (sample_count,):
        raise ValueError("split must have shape [N]")
    if value.dtype.kind not in {"U", "S"}:
        raise TypeError("split must contain fixed-width strings")
    labels = value.astype(str)
    unknown = sorted(set(labels) - set(VALID_SPLITS))
    if unknown:
        raise ValueError(f"split contains unsupported labels: {unknown}")
    return labels


def _validate_episode_split_isolation(
    task_index: np.ndarray,
    source_episode_id: np.ndarray,
    split: np.ndarray,
) -> None:
    source_values: np.ndarray
    if source_episode_id.dtype.kind in {"U", "S"}:
        source_values = source_episode_id.astype(str)
    else:
        source_values = source_episode_id

    episode_split: dict[tuple[int, object], str] = {}
    for task, source, label in zip(task_index, source_values, split, strict=True):
        source_key = source.item() if isinstance(source, np.generic) else source
        key = (int(task), source_key)
        previous = episode_split.setdefault(key, str(label))
        if previous != str(label):
            raise ValueError(
                "source physical episode crosses splits: "
                f"task_index={key[0]}, source_episode_id={key[1]!r}, "
                f"splits={previous!r}/{str(label)!r}"
            )


def _validate_task_instance_split_isolation(
    task_index: np.ndarray,
    task_instance_index: np.ndarray,
    split: np.ndarray,
) -> None:
    instance_split: dict[tuple[int, int], str] = {}
    for task, instance, label in zip(
        task_index, task_instance_index, split, strict=True
    ):
        key = (int(task), int(instance))
        previous = instance_split.setdefault(key, str(label))
        if previous != str(label):
            raise ValueError(
                "MetaWorld task instance crosses splits: "
                f"task_index={key[0]}, task_instance_index={key[1]}, "
                f"splits={previous!r}/{str(label)!r}"
            )


def validate_a1_arrays(arrays: Mapping[str, np.ndarray]) -> A1ArrayAxes:
    """Validate the complete lightweight A1 NPZ array contract.

    Episode identity is ``(task_index, source_episode_id)`` because source
    episode identifiers may be reused by different tasks.  Every hazard and
    branch from one such physical episode must remain in one split.  The
    underlying MetaWorld instance ``(task_index, task_instance_index)`` must
    also remain in one split, independently of episode identifiers.
    """

    if not isinstance(arrays, Mapping):
        raise TypeError("A1 arrays must be provided as a mapping")
    _exact_keys(arrays)
    if any(not isinstance(value, np.ndarray) for value in arrays.values()):
        raise TypeError("every A1 value must already be a numpy ndarray")

    future_action = arrays["future_action"]
    if future_action.ndim != 4 or future_action.shape[1] != CANDIDATE_COUNT:
        raise ValueError("future_action must have shape [N,8,P,20]")
    sample_count = int(future_action.shape[0])
    if sample_count <= 0:
        raise ValueError("A1 dataset must not be empty")
    horizon = int(future_action.shape[2])
    if horizon < MIN_HORIZON or future_action.shape[3] != ACTION_DIM:
        raise ValueError("future_action must have shape [N,8,P,20] with P>=8")

    expected_shapes = {
        "history_embedding": (
            sample_count,
            HISTORY_STEPS,
            KEYPOINT_COUNT,
            EMBEDDING_DIM,
        ),
        "history_action": (sample_count, HISTORY_STEPS - 1, ACTION_DIM),
        # Audit-only scripted future center; model inputs must not consume it.
        "center_action": (sample_count, horizon, ACTION_DIM),
        "target_embedding": (
            sample_count,
            CANDIDATE_COUNT,
            horizon,
            KEYPOINT_COUNT,
            EMBEDDING_DIM,
        ),
    }
    for name, expected in expected_shapes.items():
        if arrays[name].shape != expected:
            raise ValueError(f"{name} must have shape {expected}")

    for name in (
        "history_embedding",
        "history_action",
        "center_action",
        "future_action",
        "target_embedding",
    ):
        _finite_numeric(arrays[name], name)

    _integer_vector(arrays["task_index"], "task_index", sample_count)
    _integer_vector(
        arrays["task_instance_index"], "task_instance_index", sample_count
    )
    task_instance_index = arrays["task_instance_index"]
    if not np.logical_and(
        task_instance_index >= 0,
        task_instance_index < TASK_INSTANCE_COUNT,
    ).all():
        raise ValueError("task_instance_index must be in the inclusive range 0..49")
    _identifier_vector(
        arrays["source_episode_id"], "source_episode_id", sample_count
    )
    _integer_vector(arrays["hazard_id"], "hazard_id", sample_count)
    split = _split_vector(arrays["split"], sample_count)
    _validate_episode_split_isolation(
        arrays["task_index"], arrays["source_episode_id"], split
    )
    _validate_task_instance_split_isolation(
        arrays["task_index"], task_instance_index, split
    )

    return A1ArrayAxes(
        sample_count=sample_count,
        history_steps=HISTORY_STEPS,
        candidate_count=CANDIDATE_COUNT,
        horizon=horizon,
        action_dim=ACTION_DIM,
        keypoint_count=KEYPOINT_COUNT,
        embedding_dim=EMBEDDING_DIM,
    )


def save_a1_npz(
    path: str | os.PathLike[str],
    arrays: Mapping[str, np.ndarray],
    *,
    compressed: bool = True,
) -> Path:
    """Validate and atomically replace one NPZ, without sidecar metadata."""

    validate_a1_arrays(arrays)
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    writer = np.savez_compressed if compressed else np.savez
    try:
        with os.fdopen(descriptor, "wb") as handle:
            writer(handle, **{name: arrays[name] for name in NPZ_KEYS})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_a1_npz(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """Load owned, non-pickle arrays and validate the complete A1 contract."""

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"A1 NPZ does not exist: {source}")
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    validate_a1_arrays(arrays)
    return arrays


def _bounds(
    value: float | Sequence[float] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 0:
        result = np.full(RAW_ACTION_DIM, float(result), dtype=np.float64)
    if result.shape != (RAW_ACTION_DIM,):
        raise ValueError(f"{name} must be scalar or have shape [4]")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _center_and_horizon(
    scripted_center_raw: np.ndarray,
    *,
    horizon: int | None,
    low: np.ndarray,
    high: np.ndarray,
) -> tuple[np.ndarray, int]:
    center_array = np.asarray(scripted_center_raw)
    if center_array.ndim != 2 or center_array.shape[1:] != (RAW_ACTION_DIM,):
        raise ValueError("scripted_center_raw must have exact shape [P*5,4]")
    if center_array.shape[0] % RAW_STEPS_PER_HORIZON != 0:
        raise ValueError("scripted_center_raw length must equal P*5 exactly")
    inferred_horizon = center_array.shape[0] // RAW_STEPS_PER_HORIZON
    if inferred_horizon < MIN_HORIZON:
        raise ValueError("scripted_center_raw requires P>=8 (at least 40 raw actions)")
    if horizon is not None:
        if isinstance(horizon, (bool, np.bool_)) or not isinstance(
            horizon, (int, np.integer)
        ):
            raise TypeError("horizon must be an integer")
        if int(horizon) != inferred_horizon:
            raise ValueError(
                "scripted_center_raw must supply exactly horizon*5 raw actions"
            )
    if bool(np.any(low >= high)):
        raise ValueError("every action_low value must be below action_high")
    center = np.asarray(center_array, dtype=np.float64)
    if not np.issubdtype(center_array.dtype, np.number) or not np.isfinite(center).all():
        raise ValueError("scripted_center_raw must contain finite numeric data")
    if bool(np.any(center < low) or np.any(center > high)):
        raise ValueError("scripted_center_raw must already satisfy action bounds")
    return center, inferred_horizon


def _transition_axes(horizon: int, direction_seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(direction_seed))
    raw = rng.standard_normal((horizon * MOTION_DIM, AXIS_COUNT))
    basis, _ = np.linalg.qr(raw, mode="reduced")

    # Canonicalize QR's sign ambiguity independently of downstream outcomes.
    for index in range(AXIS_COUNT):
        pivot = int(np.argmax(np.abs(basis[:, index])))
        if basis[pivot, index] < 0.0:
            basis[:, index] *= -1.0
    if not np.allclose(
        basis.T @ basis,
        np.eye(AXIS_COUNT),
        rtol=0.0,
        atol=ORTHONORMAL_ABSOLUTE_TOLERANCE,
    ):
        raise RuntimeError("A1 QR action directions are not orthonormal")
    return basis.T.reshape(AXIS_COUNT, horizon, MOTION_DIM)


def _solve_clipped_direction(
    center_motion: np.ndarray,
    direction: np.ndarray,
    low_motion: np.ndarray,
    high_motion: np.ndarray,
) -> np.ndarray:
    destination = np.where(direction > 0.0, high_motion, low_motion)
    maximum_delta = np.where(direction == 0.0, 0.0, destination - center_motion)
    maximum_rms = float(np.sqrt(np.mean(np.square(maximum_delta))))
    if maximum_rms + RMS_ABSOLUTE_TOLERANCE < TARGET_MOTION_RMS:
        raise ValueError(
            "fixed motion RMS 0.15 is unreachable after clipping "
            f"(maximum {maximum_rms:.8f})"
        )

    nonzero = np.abs(direction) > np.finfo(np.float64).eps
    saturation = np.zeros_like(direction)
    saturation[nonzero] = np.abs(maximum_delta[nonzero] / direction[nonzero])
    high_scale = float(np.max(saturation))
    if not np.isfinite(high_scale) or high_scale <= 0.0:
        raise ValueError("action direction has no reachable motion")

    def realize(scale: float) -> np.ndarray:
        return np.clip(
            center_motion + scale * direction,
            low_motion,
            high_motion,
        )

    saturated = realize(high_scale * (1.0 + 8.0 * np.finfo(np.float64).eps))
    saturated_rms = float(np.sqrt(np.mean(np.square(saturated - center_motion))))
    if saturated_rms + RMS_ABSOLUTE_TOLERANCE < TARGET_MOTION_RMS:
        raise ValueError("fixed motion RMS 0.15 is numerically unreachable")

    low_scale = 0.0
    for _ in range(BISECTION_STEPS):
        midpoint = 0.5 * (low_scale + high_scale)
        realized = realize(midpoint)
        realized_rms = float(
            np.sqrt(np.mean(np.square(realized - center_motion)))
        )
        if realized_rms < TARGET_MOTION_RMS:
            low_scale = midpoint
        else:
            high_scale = midpoint
    return realize(high_scale)


def motion_delta_rms(
    action_branches_raw: np.ndarray,
    scripted_center_raw: np.ndarray,
) -> np.ndarray:
    """Return each branch's RMS delta over all raw motion coordinates."""

    branches = np.asarray(action_branches_raw, dtype=np.float64)
    center = np.asarray(scripted_center_raw, dtype=np.float64)
    if branches.ndim == 2:
        branches = branches[None, ...]
    if center.ndim != 2 or center.shape[1:] != (RAW_ACTION_DIM,):
        raise ValueError("scripted_center_raw must have shape [P*5,4]")
    if branches.ndim != 3 or branches.shape[1:] != center.shape:
        raise ValueError("action_branches_raw must have shape [C,P*5,4]")
    delta = branches[:, :, :MOTION_DIM] - center[None, :, :MOTION_DIM]
    return np.sqrt(np.mean(np.square(delta), axis=(1, 2)))


def validate_action_branches(
    action_branches_raw: np.ndarray,
    scripted_center_raw: np.ndarray,
    *,
    horizon: int | None = None,
    action_low: float | Sequence[float] | np.ndarray = -1.0,
    action_high: float | Sequence[float] | np.ndarray = 1.0,
) -> ActionBranchValidation:
    """Validate shape, bounds, gripper identity, radius, and diversity."""

    low = _bounds(action_low, name="action_low")
    high = _bounds(action_high, name="action_high")
    center, inferred_horizon = _center_and_horizon(
        scripted_center_raw,
        horizon=horizon,
        low=low,
        high=high,
    )
    branches = np.asarray(action_branches_raw)
    expected_shape = (
        CANDIDATE_COUNT,
        inferred_horizon * RAW_STEPS_PER_HORIZON,
        RAW_ACTION_DIM,
    )
    if branches.shape != expected_shape:
        raise ValueError(
            f"action_branches_raw must have shape {expected_shape}, got {branches.shape}"
        )
    _finite_numeric(branches, "action_branches_raw")
    branches64 = np.asarray(branches, dtype=np.float64)
    if bool(np.any(branches64 < low) or np.any(branches64 > high)):
        raise ValueError("action branches would change under executor clipping")
    expected_gripper = np.broadcast_to(
        np.asarray(center[:, MOTION_DIM], dtype=branches.dtype)[None, :],
        (CANDIDATE_COUNT, center.shape[0]),
    )
    if not np.array_equal(branches[:, :, MOTION_DIM], expected_gripper):
        raise ValueError("every branch must preserve scripted expert gripper commands")

    rms = motion_delta_rms(branches64, center)
    if not np.allclose(
        rms,
        TARGET_MOTION_RMS,
        rtol=0.0,
        atol=RMS_ABSOLUTE_TOLERANCE,
    ):
        raise ValueError("branch motion RMS differs from the fixed 0.15 radius")

    realized_delta = branches64[:, :, :MOTION_DIM] - center[None, :, :MOTION_DIM]
    flattened = realized_delta.reshape(CANDIDATE_COUNT, -1)
    norms = np.linalg.norm(flattened, axis=1)
    pairwise_rms: list[float] = []
    pairwise_cosine: list[float] = []
    for left_index in range(CANDIDATE_COUNT):
        for right_index in range(left_index + 1, CANDIDATE_COUNT):
            difference = flattened[left_index] - flattened[right_index]
            pairwise_rms.append(float(np.sqrt(np.mean(np.square(difference)))))
            pairwise_cosine.append(
                float(
                    np.dot(flattened[left_index], flattened[right_index])
                    / (norms[left_index] * norms[right_index])
                )
            )
    minimum_pairwise = min(pairwise_rms)
    maximum_cosine = max(pairwise_cosine)
    if minimum_pairwise < MIN_PAIRWISE_MOTION_RMS:
        raise ValueError("action branches contain duplicate or degenerate directions")
    if maximum_cosine > MAX_PAIRWISE_MOTION_COSINE:
        raise ValueError("action branch directions are insufficiently diverse")
    return ActionBranchValidation(
        horizon=inferred_horizon,
        realized_motion_rms=tuple(float(value) for value in rms),
        minimum_pairwise_motion_rms=minimum_pairwise,
        maximum_pairwise_motion_cosine=maximum_cosine,
    )


def build_anonymous_action_branches(
    scripted_center_raw: np.ndarray,
    *,
    direction_seed: int,
    permutation_seed: int,
    horizon: int | None = None,
    action_low: float | Sequence[float] | np.ndarray = -1.0,
    action_high: float | Sequence[float] | np.ndarray = 1.0,
) -> np.ndarray:
    """Build eight anonymous antithetic branches around a supplied center.

    Four QR directions in ``[P,3]`` transition space are expanded to five raw
    controller actions per horizon step and evaluated with both signs.  Each
    signed direction is independently scaled after clipping so its executed
    motion delta has RMS 0.15.  The returned slot order is independently
    permuted and contains no direction or sign labels.
    """

    low = _bounds(action_low, name="action_low")
    high = _bounds(action_high, name="action_high")
    center, inferred_horizon = _center_and_horizon(
        scripted_center_raw,
        horizon=horizon,
        low=low,
        high=high,
    )
    axes = _transition_axes(inferred_horizon, direction_seed)
    canonical: list[np.ndarray] = []
    for axis in axes:
        for sign in (1, -1):
            direction = np.repeat(
                float(sign) * axis,
                RAW_STEPS_PER_HORIZON,
                axis=0,
            )
            action = center.copy()
            action[:, :MOTION_DIM] = _solve_clipped_direction(
                center[:, :MOTION_DIM],
                direction,
                low[:MOTION_DIM],
                high[:MOTION_DIM],
            )
            canonical.append(action)

    permutation = np.random.default_rng(int(permutation_seed)).permutation(
        CANDIDATE_COUNT
    )
    branches = np.stack(canonical)[permutation].astype(np.float32, copy=False)
    branches[:, :, MOTION_DIM] = np.asarray(
        center[:, MOTION_DIM], dtype=np.float32
    )[None, :]
    validate_action_branches(
        branches,
        np.asarray(center, dtype=np.float32),
        horizon=inferred_horizon,
        action_low=low,
        action_high=high,
    )
    return branches


def reshape_action_branches(
    action_branches_raw: np.ndarray,
    *,
    horizon: int | None = None,
) -> np.ndarray:
    """Reshape exact ``[8,P*5,4]`` raw branches to ``[8,P,20]`` blocks."""

    branches = np.asarray(action_branches_raw)
    if (
        branches.ndim != 3
        or branches.shape[0] != CANDIDATE_COUNT
        or branches.shape[2] != RAW_ACTION_DIM
        or branches.shape[1] % RAW_STEPS_PER_HORIZON != 0
    ):
        raise ValueError("action_branches_raw must have exact shape [8,P*5,4]")
    inferred_horizon = branches.shape[1] // RAW_STEPS_PER_HORIZON
    if inferred_horizon < MIN_HORIZON:
        raise ValueError("action_branches_raw requires P>=8")
    if horizon is not None:
        if isinstance(horizon, (bool, np.bool_)) or not isinstance(
            horizon, (int, np.integer)
        ):
            raise TypeError("horizon must be an integer")
        if int(horizon) != inferred_horizon:
            raise ValueError("action_branches_raw must contain exactly horizon*5 actions")
    _finite_numeric(branches, "action_branches_raw")
    return branches.reshape(CANDIDATE_COUNT, inferred_horizon, ACTION_DIM)


__all__ = [
    "ACTION_DIM",
    "ANONYMOUS_BRANCH_NAMES",
    "AXIS_COUNT",
    "ActionBranchValidation",
    "A1ArrayAxes",
    "BISECTION_STEPS",
    "CANDIDATE_COUNT",
    "EMBEDDING_DIM",
    "FORMAL_TASK_INSTANCE_COUNT",
    "HISTORY_STEPS",
    "KEYPOINT_COUNT",
    "MAX_PAIRWISE_MOTION_COSINE",
    "MIN_HORIZON",
    "MIN_PAIRWISE_MOTION_RMS",
    "MOTION_DIM",
    "NPZ_KEYS",
    "RAW_ACTION_DIM",
    "RAW_STEPS_PER_HORIZON",
    "RMS_ABSOLUTE_TOLERANCE",
    "SMOKE_TASK_INSTANCE_COUNT",
    "TASK_INSTANCE_COUNT",
    "TARGET_MOTION_RMS",
    "VALID_SPLITS",
    "build_anonymous_action_branches",
    "load_a1_npz",
    "motion_delta_rms",
    "reshape_action_branches",
    "save_a1_npz",
    "validate_a1_arrays",
    "validate_action_branches",
]
