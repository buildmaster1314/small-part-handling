"""Pure control contracts for the SO-101 receding-horizon runtime.

This module deliberately has no LeRobot or web-server dependency.  It owns the
small state machine and fail-closed checks shared by replay tests and the real
hardware runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from fcpg.so101_retrieval_mpc import SO101_POSITION_LOWER, SO101_POSITION_UPPER


class ManipulationPhase(str, Enum):
    TRAJECTORY = "trajectory"
    APPROACH = "approach"
    GRASP = "grasp"
    TRANSPORT = "transport"
    RELEASE = "release"
    DONE = "done"
    FAULT = "fault"


@dataclass(frozen=True)
class PhaseThresholds:
    pickup_near_px: float
    drop_near_px: float
    pickup_near_m: float
    drop_near_m: float
    gripper_closed: float
    gripper_open: float
    grasp_roi_px: float
    confirmations: int = 2
    # Aperture at/below which a closed gripper is considered EMPTY (bottomed
    # out near the physical stop). A real grasp stalls against the object
    # well above this; attachment evidence requires the stall.
    gripper_empty_closed: float = 0.0

    def __post_init__(self) -> None:
        finite_positive = (
            self.pickup_near_px,
            self.drop_near_px,
            self.pickup_near_m,
            self.drop_near_m,
            self.grasp_roi_px,
        )
        if any(not np.isfinite(value) or value <= 0 for value in finite_positive):
            raise ValueError("phase thresholds must be finite and positive")
        if not np.isfinite(self.gripper_closed) or not np.isfinite(self.gripper_open):
            raise ValueError("gripper thresholds must be finite")
        if self.gripper_closed >= self.gripper_open:
            raise ValueError("gripper_closed must be lower than gripper_open")
        if (
            not np.isfinite(self.gripper_empty_closed)
            or self.gripper_empty_closed >= self.gripper_closed
        ):
            raise ValueError(
                "gripper_empty_closed must be finite and below gripper_closed"
            )
        if self.confirmations <= 0:
            raise ValueError("confirmations must be positive")


@dataclass(frozen=True)
class PhaseEvidence:
    gripper_position: float
    gripper_to_pickup_px: float | None
    gripper_to_drop_px: float | None
    gripper_to_pickup_m: float | None
    gripper_to_drop_m: float | None
    wrist_pickup_distance_px: float | None
    pickup_visible: bool
    drop_zone_visible: bool
    pickup_inside_drop_zone: bool

    def __post_init__(self) -> None:
        if not np.isfinite(self.gripper_position):
            raise ValueError("gripper_position must be finite")
        for name in (
            "gripper_to_pickup_px",
            "gripper_to_drop_px",
            "gripper_to_pickup_m",
            "gripper_to_drop_m",
            "wrist_pickup_distance_px",
        ):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")


class PhaseTracker:
    """Causal manipulation phase tracker using only deployable evidence."""

    def __init__(
        self,
        thresholds: PhaseThresholds,
        *,
        initial: ManipulationPhase = ManipulationPhase.APPROACH,
    ) -> None:
        self.thresholds = thresholds
        self.phase = initial
        self._candidate_phase: ManipulationPhase | None = None
        self._candidate_count = 0

    def _confirm(self, candidate: ManipulationPhase | None) -> None:
        if candidate is None:
            self._candidate_phase = None
            self._candidate_count = 0
            return
        if candidate == self._candidate_phase:
            self._candidate_count += 1
        else:
            self._candidate_phase = candidate
            self._candidate_count = 1
        if self._candidate_count >= self.thresholds.confirmations:
            self.phase = candidate
            self._candidate_phase = None
            self._candidate_count = 0

    def update(self, evidence: PhaseEvidence) -> ManipulationPhase:
        if self.phase in {ManipulationPhase.DONE, ManipulationPhase.FAULT}:
            return self.phase

        candidate: ManipulationPhase | None = None
        if self.phase == ManipulationPhase.APPROACH:
            if (
                evidence.pickup_visible
                and evidence.gripper_to_pickup_px is not None
                and evidence.gripper_to_pickup_px
                <= self.thresholds.pickup_near_px
                and evidence.gripper_to_pickup_m is not None
                and evidence.gripper_to_pickup_m <= self.thresholds.pickup_near_m
            ):
                candidate = ManipulationPhase.GRASP
        elif self.phase == ManipulationPhase.GRASP:
            attached = (
                evidence.pickup_visible
                and evidence.wrist_pickup_distance_px is not None
                and evidence.wrist_pickup_distance_px
                <= self.thresholds.grasp_roi_px
                and evidence.gripper_position <= self.thresholds.gripper_closed
                # A close that bottoms out near the physical stop closed on
                # air; only a stall ABOVE the empty band proves something is
                # between the jaws.
                and evidence.gripper_position
                > self.thresholds.gripper_empty_closed
            )
            if attached:
                candidate = ManipulationPhase.TRANSPORT
        elif self.phase == ManipulationPhase.TRANSPORT:
            if (
                evidence.drop_zone_visible
                and evidence.gripper_to_drop_px is not None
                and evidence.gripper_to_drop_px <= self.thresholds.drop_near_px
                and evidence.gripper_to_drop_m is not None
                and evidence.gripper_to_drop_m <= self.thresholds.drop_near_m
            ):
                candidate = ManipulationPhase.RELEASE
        elif self.phase == ManipulationPhase.RELEASE:
            if (
                evidence.pickup_inside_drop_zone
                and evidence.gripper_position >= self.thresholds.gripper_open
            ):
                candidate = ManipulationPhase.DONE

        self._confirm(candidate)
        return self.phase

    def regress_to_approach(self) -> None:
        """Retreat from GRASP to re-servo when the close pose is misaligned.

        The only sanctioned backward edge: GRASP freezes the arm, so when the
        jaws are open and the object is no longer under them (it moved, or a
        reopen retry needs re-aiming) alignment can only be recovered by
        going back to APPROACH.
        """

        if self.phase == ManipulationPhase.GRASP:
            self.phase = ManipulationPhase.APPROACH
            self._candidate_phase = None
            self._candidate_count = 0

    def fault(self) -> None:
        self.phase = ManipulationPhase.FAULT
        self._candidate_phase = None
        self._candidate_count = 0

    def reset(self) -> None:
        self.phase = ManipulationPhase.APPROACH
        self._candidate_phase = None
        self._candidate_count = 0

    def status(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "pending_phase": (
                None if self._candidate_phase is None else self._candidate_phase.value
            ),
            "pending_confirmations": self._candidate_count,
            "required_confirmations": self.thresholds.confirmations,
        }


def target_role_for_phase(phase: ManipulationPhase) -> str | None:
    if phase in {ManipulationPhase.APPROACH, ManipulationPhase.GRASP}:
        return "pickup"
    if phase in {ManipulationPhase.TRANSPORT, ManipulationPhase.RELEASE}:
        return "drop_zone"
    return None


# Retained as an API alias for older replay tools.  Values are generic roles,
# never color or object-class names.
target_kind_for_phase = target_role_for_phase


def candidate_phase_mask(
    prefixes: np.ndarray,
    phase: ManipulationPhase,
    thresholds: PhaseThresholds,
    *,
    current_gripper: float | None = None,
    recovery_below: float | None = None,
) -> np.ndarray:
    """Return candidates whose gripper evolution matches the active phase."""

    values = np.asarray(prefixes, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 6 or values.shape[1] == 0:
        raise ValueError("prefixes must have shape [K,R,6]")
    if not np.all(np.isfinite(values)):
        raise ValueError("prefixes must be finite")
    initial = values[:, 0, -1]
    terminal = values[:, -1, -1]
    minimum = np.min(values[:, :, -1], axis=1)
    maximum = np.max(values[:, :, -1], axis=1)
    delta = terminal - initial

    if phase == ManipulationPhase.APPROACH:
        # While approaching, never begin a closing motion.  A prefix may open
        # gradually from a nearly closed live pose, so direction is more
        # reliable than an absolute aperture threshold.
        return delta >= -0.5
    if phase == ManipulationPhase.GRASP:
        closing = (minimum <= thresholds.gripper_closed) & (delta <= -0.5)
        if (
            current_gripper is not None
            and recovery_below is not None
            and current_gripper <= recovery_below
        ):
            # The live gripper is truly at the physical stop (empty-closed):
            # no candidate can demonstrate further closing, so admit opening
            # recovery motions instead of deadlocking the phase. A grasp
            # stall against an object sits well above this threshold and
            # must never admit opening — that would drop the object.
            return closing | (delta >= 0.5)
        return closing
    if phase == ManipulationPhase.TRANSPORT:
        return (maximum <= thresholds.gripper_open) & (np.abs(delta) <= 5.0)
    if phase == ManipulationPhase.RELEASE:
        return (maximum >= thresholds.gripper_open) & (delta >= 0.5)
    return np.zeros(len(values), dtype=bool)


def phase_prediction_bounds_mask(
    predicted_states: np.ndarray,
    phase: ManipulationPhase,
    *,
    margin: float,
) -> np.ndarray:
    """Check LeWM forecasts without conflating approach-arm and gripper safety.

    During approach, the absolute command prefix already checks all six hard
    limits and the gripper is not the controlled motion. A small forecast error
    in the LeWM gripper head must not reject an otherwise bounded arm path.
    """

    values = np.asarray(predicted_states, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 6:
        raise ValueError("predicted_states must have shape [K,H,6]")
    if not np.all(np.isfinite(values)):
        return np.zeros(len(values), dtype=bool)
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("prediction bound margin must be finite and non-negative")
    dimensions = 5 if phase == ManipulationPhase.APPROACH else 6
    lower = SO101_POSITION_LOWER[:dimensions] - margin
    upper = SO101_POSITION_UPPER[:dimensions] + margin
    checked = values[..., :dimensions]
    return np.all(
        (checked >= lower[None, None]) & (checked <= upper[None, None]),
        axis=(1, 2),
    )


@dataclass(frozen=True)
class CommandChunkCheck:
    safe: bool
    reasons: tuple[str, ...]
    first_jump: float
    maximum_step_jump: float


def validate_command_chunk(
    chunk: np.ndarray,
    current_state: np.ndarray,
    *,
    expected_commands: int = 5,
    max_first_jump: float = 8.0,
    max_step_jump: float = 8.0,
    limit_guard: float = 3.0,
    outward_tolerance: float = 0.0,
) -> CommandChunkCheck:
    values = np.asarray(chunk, dtype=np.float32)
    state = np.asarray(current_state, dtype=np.float32)
    if values.shape != (expected_commands, 6):
        raise ValueError(f"command chunk must have shape [{expected_commands},6]")
    if state.shape != (6,):
        raise ValueError("current_state must have shape [6]")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(state)):
        raise ValueError("command chunk and current state must be finite")
    if min(max_first_jump, max_step_jump, limit_guard) <= 0:
        raise ValueError("command safety limits must be positive")
    if not np.isfinite(outward_tolerance) or outward_tolerance < 0:
        raise ValueError("outward_tolerance must be finite and non-negative")

    previous = np.concatenate((state[None], values[:-1]), axis=0)
    jumps = np.max(np.abs(values - previous), axis=1)
    first_jump = float(jumps[0])
    maximum_step = float(np.max(jumps))
    reasons: list[str] = []
    if np.any(values < SO101_POSITION_LOWER) or np.any(values > SO101_POSITION_UPPER):
        reasons.append("hard_position_bounds")
    if first_jump > max_first_jump:
        reasons.append("first_command_jump")
    if maximum_step > max_step_jump:
        reasons.append("prefix_step_jump")

    lower_near = state <= SO101_POSITION_LOWER + limit_guard
    upper_near = state >= SO101_POSITION_UPPER - limit_guard
    first_delta = values[0] - state
    if np.any(lower_near & (first_delta < -outward_tolerance)) or np.any(
        upper_near & (first_delta > outward_tolerance)
    ):
        reasons.append("moving_outward_at_joint_limit")
    return CommandChunkCheck(
        safe=not reasons,
        reasons=tuple(reasons),
        first_jump=first_jump,
        maximum_step_jump=maximum_step,
    )


@dataclass(frozen=True)
class PlanningRoundGuard:
    round_id: str
    created_monotonic_s: float
    state: np.ndarray
    target_px: np.ndarray
    target_kind: str

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=np.float32)
        target = np.asarray(self.target_px, dtype=np.float32)
        if not self.round_id:
            raise ValueError("round_id cannot be empty")
        if not np.isfinite(self.created_monotonic_s):
            raise ValueError("created_monotonic_s must be finite")
        if state.shape != (6,) or target.shape != (2,):
            raise ValueError("round guard state/target shapes are invalid")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(target)):
            raise ValueError("round guard state/target must be finite")

    def validate_live(
        self,
        *,
        round_id: str,
        now_monotonic_s: float,
        live_state: np.ndarray,
        live_target_px: np.ndarray | None,
        max_age_s: float,
        max_state_drift: float,
        max_target_drift_px: float,
    ) -> dict[str, float | None]:
        if round_id != self.round_id:
            raise RuntimeError("stale planning round")
        age = float(now_monotonic_s - self.created_monotonic_s)
        if age < 0 or (max_age_s > 0 and age > max_age_s):
            raise RuntimeError(f"planning round age {age:.3f}s exceeds limit")
        state = np.asarray(live_state, dtype=np.float32)
        if state.shape != (6,):
            raise ValueError("live state/target shapes are invalid")
        state_drift = float(np.max(np.abs(state - np.asarray(self.state))))
        if state_drift > max_state_drift:
            raise RuntimeError(
                f"live state drift {state_drift:.3f} exceeds {max_state_drift:.3f}"
            )
        if live_target_px is None:
            # No live target measurement exists for this plan; the caller must
            # record an explicit skip rather than compare the plan's own
            # target to itself.
            return {
                "age_s": age,
                "state_drift": state_drift,
                "target_drift_px": None,
            }
        target = np.asarray(live_target_px, dtype=np.float32)
        if target.shape != (2,):
            raise ValueError("live state/target shapes are invalid")
        target_drift = float(np.linalg.norm(target - np.asarray(self.target_px)))
        if target_drift > max_target_drift_px:
            raise RuntimeError(
                f"live target drift {target_drift:.3f}px exceeds "
                f"{max_target_drift_px:.3f}px"
            )
        return {
            "age_s": age,
            "state_drift": state_drift,
            "target_drift_px": target_drift,
        }


__all__ = [
    "CommandChunkCheck",
    "ManipulationPhase",
    "PhaseEvidence",
    "PhaseThresholds",
    "PhaseTracker",
    "PlanningRoundGuard",
    "candidate_phase_mask",
    "phase_prediction_bounds_mask",
    "target_kind_for_phase",
    "target_role_for_phase",
    "validate_command_chunk",
]
