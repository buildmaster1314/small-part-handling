"""Leak-resistant equal-norm action candidates for Counterfactual v3.

The deployment-facing candidate bank contains anonymous slots only.  Axis
identities and the slot permutation live in a separate audit object so a
scorer cannot recover candidate semantics from names or metadata.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from typing import Sequence

import numpy as np


RAW_ACTION_STEPS = 25
ACTION_DIM = 4
MOTION_DIM = 3
TRANSITIONS = 5
STEPS_PER_TRANSITION = 5
AXIS_COUNT = 4
CANDIDATE_COUNT = 2 * AXIS_COUNT
TARGET_MOTION_RMS = 0.15

# These thresholds are protocol constants, not validation-set hyperparameters.
RMS_ABSOLUTE_TOLERANCE = 2.0e-6
ORTHONORMAL_ABSOLUTE_TOLERANCE = 1.0e-10
MIN_PAIRWISE_MOTION_RMS = 0.90 * TARGET_MOTION_RMS
MAX_PAIRWISE_MOTION_COSINE = 0.60
BISECTION_STEPS = 96
MAX_SLOT_PRIOR_ABS_SPEARMAN = 0.10
MAX_SLOT_PRIOR_NORMALIZED_REGRET_IMPROVEMENT = 0.02


def _readonly_float32(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.float32, copy=True, order="C")
    result.setflags(write=False)
    return result


def _bounds(
    value: float | Sequence[float] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 0:
        result = np.full(ACTION_DIM, float(result), dtype=np.float64)
    if result.shape != (ACTION_DIM,):
        raise ValueError(f"{name} must be scalar or have shape {(ACTION_DIM,)}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _policy_center(
    value: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    center = np.asarray(value, dtype=np.float64)
    if center.shape != (RAW_ACTION_STEPS, ACTION_DIM):
        raise ValueError(
            "policy_action must have shape "
            f"{(RAW_ACTION_STEPS, ACTION_DIM)}, got {tuple(center.shape)}"
        )
    if not np.isfinite(center).all():
        raise ValueError("policy_action must be finite")
    if bool(np.any(low >= high)):
        raise ValueError("every action_low value must be below action_high")
    clipped = np.clip(center, low, high)
    if not np.array_equal(center, clipped):
        raise ValueError("policy_action must already equal executor clipping")
    return center


def _direction_sha256(value: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


@dataclass(frozen=True, slots=True)
class AnonymousCandidateBank:
    """Deployment-visible anonymous candidate actions.

    No axis, sign, perturbation kind, or permutation identifier is present in
    this object.  Scorers may consume ``names`` and ``actions`` only.
    """

    names: tuple[str, ...]
    actions: np.ndarray

    def __post_init__(self) -> None:
        names = tuple(str(value) for value in self.names)
        expected_names = tuple(f"candidate_{index}" for index in range(CANDIDATE_COUNT))
        if names != expected_names:
            raise ValueError("candidate names must be the fixed anonymous slot names")
        actions = np.asarray(self.actions)
        if actions.shape != (
            CANDIDATE_COUNT,
            RAW_ACTION_STEPS,
            ACTION_DIM,
        ):
            raise ValueError("candidate actions must have shape [8,25,4]")
        if not np.isfinite(actions).all():
            raise ValueError("candidate actions must be finite")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "actions", _readonly_float32(actions))


@dataclass(frozen=True, slots=True)
class EqualNormCandidateAudit:
    """Collector-only identities that must never be passed to a scorer."""

    slot_to_canonical_direction: tuple[int, ...]
    canonical_direction_sha256: tuple[str, ...]
    realized_motion_rms: tuple[float, ...]
    direction_seed: int
    permutation_seed: int

    def __post_init__(self) -> None:
        permutation = tuple(int(value) for value in self.slot_to_canonical_direction)
        if sorted(permutation) != list(range(CANDIDATE_COUNT)):
            raise ValueError("slot_to_canonical_direction must be a permutation of 0..7")
        hashes = tuple(str(value) for value in self.canonical_direction_sha256)
        if len(hashes) != CANDIDATE_COUNT or len(set(hashes)) != CANDIDATE_COUNT:
            raise ValueError("canonical direction hashes must be eight unique values")
        if any(len(value) != 64 for value in hashes):
            raise ValueError("canonical direction hashes must be SHA-256 hex digests")
        try:
            for value in hashes:
                int(value, 16)
        except ValueError as exc:
            raise ValueError(
                "canonical direction hashes must be SHA-256 hex digests"
            ) from exc
        rms = tuple(float(value) for value in self.realized_motion_rms)
        if len(rms) != CANDIDATE_COUNT or not np.isfinite(rms).all():
            raise ValueError("realized_motion_rms must contain eight finite values")
        if any(
            abs(value - TARGET_MOTION_RMS) > RMS_ABSOLUTE_TOLERANCE
            for value in rms
        ):
            raise ValueError("audit contains an off-radius candidate")
        object.__setattr__(self, "slot_to_canonical_direction", permutation)
        object.__setattr__(self, "canonical_direction_sha256", hashes)
        object.__setattr__(self, "realized_motion_rms", rms)
        object.__setattr__(self, "direction_seed", int(self.direction_seed))
        object.__setattr__(self, "permutation_seed", int(self.permutation_seed))


@dataclass(frozen=True, slots=True)
class EqualNormCandidateValidation:
    """Fail-closed validation measurements for one public candidate bank."""

    realized_motion_rms: tuple[float, ...]
    minimum_pairwise_motion_rms: float
    maximum_pairwise_motion_cosine: float
    executor_clip_identity: bool


def public_candidate_fields() -> tuple[str, ...]:
    """Return the complete scorer-visible schema for audit assertions."""

    return tuple(field.name for field in fields(AnonymousCandidateBank))


def transition_axes(direction_seed: int) -> np.ndarray:
    """Generate four deterministic orthonormal axes in 5x3 transition space."""

    rng = np.random.default_rng(int(direction_seed))
    raw = rng.standard_normal((TRANSITIONS * MOTION_DIM, AXIS_COUNT))
    basis, _ = np.linalg.qr(raw, mode="reduced")

    # Canonicalize QR's sign ambiguity without using downstream outcomes.
    for index in range(AXIS_COUNT):
        pivot = int(np.argmax(np.abs(basis[:, index])))
        if basis[pivot, index] < 0.0:
            basis[:, index] *= -1.0
    gram = basis.T @ basis
    if not np.allclose(
        gram,
        np.eye(AXIS_COUNT),
        rtol=0.0,
        atol=ORTHONORMAL_ABSOLUTE_TOLERANCE,
    ):
        raise RuntimeError("QR candidate axes are not orthonormal")
    result = basis.T.reshape(AXIS_COUNT, TRANSITIONS, MOTION_DIM)
    result.setflags(write=False)
    return result


def _expanded_direction(axis: np.ndarray, sign: int) -> np.ndarray:
    if sign not in {-1, 1}:
        raise ValueError("direction sign must be -1 or +1")
    value = np.asarray(axis, dtype=np.float64)
    if value.shape != (TRANSITIONS, MOTION_DIM):
        raise ValueError("transition axis must have shape [5,3]")
    return np.repeat(sign * value, STEPS_PER_TRANSITION, axis=0)


def motion_delta_rms(actions: np.ndarray, policy_action: np.ndarray) -> np.ndarray:
    """Return per-candidate RMS displacement over the 25x3 motion entries."""

    candidate = np.asarray(actions, dtype=np.float64)
    center = np.asarray(policy_action, dtype=np.float64)
    if candidate.ndim == 2:
        candidate = candidate[None, ...]
    if candidate.ndim != 3 or candidate.shape[1:] != (
        RAW_ACTION_STEPS,
        ACTION_DIM,
    ):
        raise ValueError("actions must have shape [C,25,4] or [25,4]")
    if center.shape != (RAW_ACTION_STEPS, ACTION_DIM):
        raise ValueError("policy_action must have shape [25,4]")
    delta = candidate[:, :, :MOTION_DIM] - center[None, :, :MOTION_DIM]
    return np.sqrt(np.mean(np.square(delta), axis=(1, 2)))


def _solve_clipped_direction(
    center_motion: np.ndarray,
    direction: np.ndarray,
    low_motion: np.ndarray,
    high_motion: np.ndarray,
) -> np.ndarray:
    target = TARGET_MOTION_RMS
    destination = np.where(direction > 0.0, high_motion, low_motion)
    max_delta = np.where(direction == 0.0, 0.0, destination - center_motion)
    maximum_rms = float(np.sqrt(np.mean(np.square(max_delta))))
    if maximum_rms + RMS_ABSOLUTE_TOLERANCE < target:
        raise ValueError(
            "fixed RMS 0.15 candidate radius is unreachable after clipping "
            f"(maximum {maximum_rms:.8f})"
        )

    nonzero = np.abs(direction) > np.finfo(np.float64).eps
    saturation = np.zeros_like(direction)
    saturation[nonzero] = np.abs(max_delta[nonzero] / direction[nonzero])
    high_scale = float(np.max(saturation))
    if not np.isfinite(high_scale) or high_scale <= 0.0:
        raise ValueError("candidate direction has no reachable motion")

    def realize(scale: float) -> np.ndarray:
        return np.clip(
            center_motion + scale * direction,
            low_motion,
            high_motion,
        )

    saturated = realize(high_scale * (1.0 + 8.0 * np.finfo(np.float64).eps))
    saturated_rms = float(
        np.sqrt(np.mean(np.square(saturated - center_motion)))
    )
    if saturated_rms + RMS_ABSOLUTE_TOLERANCE < target:
        raise ValueError("candidate radius is numerically unreachable after clipping")

    low_scale = 0.0
    for _ in range(BISECTION_STEPS):
        midpoint = 0.5 * (low_scale + high_scale)
        realized = realize(midpoint)
        rms = float(np.sqrt(np.mean(np.square(realized - center_motion))))
        if rms < target:
            low_scale = midpoint
        else:
            high_scale = midpoint
    return realize(high_scale)


def validate_equal_norm_candidate_bank(
    bank: AnonymousCandidateBank,
    policy_action: np.ndarray,
    *,
    action_low: float | Sequence[float] | np.ndarray = -1.0,
    action_high: float | Sequence[float] | np.ndarray = 1.0,
) -> EqualNormCandidateValidation:
    """Validate radius, anonymity-relevant shape, bounds, and uniqueness."""

    if not isinstance(bank, AnonymousCandidateBank):
        raise TypeError("bank must be AnonymousCandidateBank")
    low = _bounds(action_low, name="action_low")
    high = _bounds(action_high, name="action_high")
    center = _policy_center(policy_action, low, high)
    actions = np.asarray(bank.actions, dtype=np.float64)
    clipped = np.clip(actions, low, high)
    clip_identity = bool(np.array_equal(actions, clipped))
    if not clip_identity:
        raise ValueError("candidate actions would change under executor clipping")
    expected_gripper = np.broadcast_to(
        center[None, :, MOTION_DIM],
        (CANDIDATE_COUNT, RAW_ACTION_STEPS),
    )
    if not np.array_equal(actions[:, :, MOTION_DIM], expected_gripper):
        raise ValueError("all candidates must preserve the policy gripper commands")

    rms = motion_delta_rms(actions, center)
    if not np.allclose(
        rms,
        TARGET_MOTION_RMS,
        rtol=0.0,
        atol=RMS_ABSOLUTE_TOLERANCE,
    ):
        raise ValueError("candidate motion RMS differs from the fixed 0.15 radius")

    pairwise = []
    pairwise_cosine = []
    realized_delta = actions[:, :, :MOTION_DIM] - center[None, :, :MOTION_DIM]
    flattened = realized_delta.reshape(CANDIDATE_COUNT, -1)
    flattened_norm = np.linalg.norm(flattened, axis=1)
    for left in range(CANDIDATE_COUNT):
        for right in range(left + 1, CANDIDATE_COUNT):
            delta = realized_delta[left] - realized_delta[right]
            pairwise.append(float(np.sqrt(np.mean(np.square(delta)))))
            pairwise_cosine.append(
                float(
                    np.dot(flattened[left], flattened[right])
                    / (flattened_norm[left] * flattened_norm[right])
                )
            )
    minimum = min(pairwise)
    if minimum < MIN_PAIRWISE_MOTION_RMS:
        raise ValueError("candidate bank contains duplicate or degenerate directions")
    maximum_cosine = max(pairwise_cosine)
    if maximum_cosine > MAX_PAIRWISE_MOTION_COSINE:
        raise ValueError("candidate bank directions are insufficiently separated")
    return EqualNormCandidateValidation(
        realized_motion_rms=tuple(float(value) for value in rms),
        minimum_pairwise_motion_rms=float(minimum),
        maximum_pairwise_motion_cosine=float(maximum_cosine),
        executor_clip_identity=True,
    )


def validate_slot_prior_audit(
    *,
    task_macro_spearman: float,
    normalized_regret_improvement: float,
    confidence_interval: tuple[float, float],
) -> None:
    """Apply the frozen candidate-slot shortcut gate.

    The train-only task+anonymous-slot lookup must remain near random.  Both a
    small point estimate and a confidence interval containing zero are
    required; non-finite or reversed intervals fail closed.
    """

    value = float(task_macro_spearman)
    regret_improvement = float(normalized_regret_improvement)
    low, high = (float(item) for item in confidence_interval)
    if not np.isfinite((value, regret_improvement, low, high)).all() or low > high:
        raise ValueError("slot-prior audit statistics must be finite and ordered")
    if abs(value) > MAX_SLOT_PRIOR_ABS_SPEARMAN:
        raise ValueError("candidate-slot prior exceeds the frozen Spearman threshold")
    if not low <= 0.0 <= high:
        raise ValueError("candidate-slot prior confidence interval excludes zero")
    if regret_improvement > MAX_SLOT_PRIOR_NORMALIZED_REGRET_IMPROVEMENT:
        raise ValueError(
            "candidate-slot prior exceeds the frozen normalized-regret threshold"
        )


def build_equal_norm_candidate_bank(
    policy_action: np.ndarray,
    *,
    direction_seed: int,
    permutation_seed: int,
    action_low: float | Sequence[float] | np.ndarray = -1.0,
    action_high: float | Sequence[float] | np.ndarray = 1.0,
) -> tuple[AnonymousCandidateBank, EqualNormCandidateAudit]:
    """Build eight clipped, equal-RMS candidates around one policy chunk.

    Four QR axes are generated in a 5x3 transition space.  Each transition is
    repeated for five raw controller steps and evaluated with both signs.  A
    separate bisection per signed direction compensates for action clipping so
    every *executed* motion perturbation has RMS exactly 0.15.
    """

    low = _bounds(action_low, name="action_low")
    high = _bounds(action_high, name="action_high")
    center = _policy_center(policy_action, low, high)
    axes = transition_axes(direction_seed)
    canonical_actions = []
    canonical_hashes = []
    for axis_index in range(AXIS_COUNT):
        for sign in (1, -1):
            direction = _expanded_direction(axes[axis_index], sign)
            motion = _solve_clipped_direction(
                center[:, :MOTION_DIM],
                direction,
                low[:MOTION_DIM],
                high[:MOTION_DIM],
            )
            action = center.copy()
            action[:, :MOTION_DIM] = motion
            canonical_actions.append(action)
            canonical_hashes.append(_direction_sha256(direction))

    canonical = np.stack(canonical_actions)
    permutation = np.random.default_rng(int(permutation_seed)).permutation(
        CANDIDATE_COUNT
    )
    actions = canonical[permutation].astype(np.float32, copy=False)
    names = tuple(f"candidate_{index}" for index in range(CANDIDATE_COUNT))
    bank = AnonymousCandidateBank(names=names, actions=actions)
    validation = validate_equal_norm_candidate_bank(
        bank,
        center.astype(np.float32),
        action_low=low,
        action_high=high,
    )
    audit = EqualNormCandidateAudit(
        slot_to_canonical_direction=tuple(int(value) for value in permutation),
        canonical_direction_sha256=tuple(canonical_hashes),
        realized_motion_rms=validation.realized_motion_rms,
        direction_seed=int(direction_seed),
        permutation_seed=int(permutation_seed),
    )
    return bank, audit


__all__ = [
    "ACTION_DIM",
    "AXIS_COUNT",
    "AnonymousCandidateBank",
    "BISECTION_STEPS",
    "CANDIDATE_COUNT",
    "EqualNormCandidateAudit",
    "EqualNormCandidateValidation",
    "MAX_SLOT_PRIOR_ABS_SPEARMAN",
    "MAX_PAIRWISE_MOTION_COSINE",
    "MAX_SLOT_PRIOR_NORMALIZED_REGRET_IMPROVEMENT",
    "MIN_PAIRWISE_MOTION_RMS",
    "MOTION_DIM",
    "ORTHONORMAL_ABSOLUTE_TOLERANCE",
    "RAW_ACTION_STEPS",
    "RMS_ABSOLUTE_TOLERANCE",
    "STEPS_PER_TRANSITION",
    "TARGET_MOTION_RMS",
    "TRANSITIONS",
    "build_equal_norm_candidate_bank",
    "motion_delta_rms",
    "public_candidate_fields",
    "transition_axes",
    "validate_equal_norm_candidate_bank",
    "validate_slot_prior_audit",
]
