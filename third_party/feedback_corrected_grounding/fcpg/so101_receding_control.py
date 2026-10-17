"""Fresh-observation contract for short-horizon SO-101 control.

This module deliberately contains no robot I/O.  It binds a selected H8
diagnostic forecast to the paired camera frames and FCPG grounding that
created it, then exposes only the first raw command chunk for execution.
After that chunk is consumed, both cameras and the grounding result must
advance before another plan can be authorized.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class FreshObservationRequired(RuntimeError):
    """Raised when planning or execution would reuse stale visual evidence."""


@dataclass(frozen=True)
class SO101ObservationIdentity:
    """Identity of one paired-camera observation after FCPG has completed."""

    session_id: str
    wrist_sequence: int
    column_sequence: int
    grounding_sequence: int
    wrist_timestamp_s: float
    column_timestamp_s: float

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must be non-empty")
        sequences = (
            self.wrist_sequence,
            self.column_sequence,
            self.grounding_sequence,
        )
        if any(isinstance(value, bool) or int(value) != value or value < 0 for value in sequences):
            raise ValueError("camera and grounding sequences must be non-negative integers")
        timestamps = np.asarray(
            (self.wrist_timestamp_s, self.column_timestamp_s), dtype=np.float64
        )
        if not np.all(np.isfinite(timestamps)):
            raise ValueError("camera timestamps must be finite")

    def is_strictly_newer_than(self, other: "SO101ObservationIdentity") -> bool:
        """Return true only when both cameras and FCPG have advanced."""

        return bool(
            self.session_id == other.session_id
            and self.wrist_sequence > other.wrist_sequence
            and self.column_sequence > other.column_sequence
            and self.grounding_sequence > other.grounding_sequence
            and self.wrist_timestamp_s > other.wrist_timestamp_s
            and self.column_timestamp_s > other.column_timestamp_s
        )


@dataclass(frozen=True)
class SO101RecedingPlan:
    """One selected H8 forecast tied to the observation that produced it."""

    observation: SO101ObservationIdentity
    target_kind: str
    target_xyz: np.ndarray
    candidate_name: str
    diagnostic_prefix: np.ndarray
    commands_per_round: int

    def __post_init__(self) -> None:
        if not self.target_kind:
            raise ValueError("target_kind must be non-empty")
        if not self.candidate_name:
            raise ValueError("candidate_name must be non-empty")
        if self.commands_per_round <= 0:
            raise ValueError("commands_per_round must be positive")

        target = np.asarray(self.target_xyz, dtype=np.float64)
        prefix = np.asarray(self.diagnostic_prefix, dtype=np.float32)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("target_xyz must be one finite XYZ vector")
        if (
            prefix.ndim != 2
            or prefix.shape[1] != 6
            or prefix.shape[0] < self.commands_per_round
            or not np.all(np.isfinite(prefix))
        ):
            raise ValueError(
                "diagnostic_prefix must be finite [R,6] and contain one full command chunk"
            )

        target = target.copy()
        prefix = prefix.copy()
        target.flags.writeable = False
        prefix.flags.writeable = False
        object.__setattr__(self, "target_xyz", target)
        object.__setattr__(self, "diagnostic_prefix", prefix)

    @property
    def diagnostic_horizon_commands(self) -> int:
        return int(self.diagnostic_prefix.shape[0])


class SO101RecedingHorizonGate:
    """Authorize one short command chunk per fresh paired observation."""

    def __init__(self, commands_per_round: int = 5) -> None:
        if commands_per_round <= 0:
            raise ValueError("commands_per_round must be positive")
        self.commands_per_round = int(commands_per_round)
        self._last_consumed: SO101ObservationIdentity | None = None

    @property
    def last_consumed(self) -> SO101ObservationIdentity | None:
        return self._last_consumed

    def _require_fresh(self, observation: SO101ObservationIdentity) -> None:
        if self._last_consumed is None:
            return
        if not observation.is_strictly_newer_than(self._last_consumed):
            raise FreshObservationRequired(
                "both camera frames and FCPG grounding must advance after each command chunk"
            )

    def build_plan(
        self,
        *,
        observation: SO101ObservationIdentity,
        target_kind: str,
        target_xyz: np.ndarray,
        candidate_name: str,
        diagnostic_prefix: np.ndarray,
    ) -> SO101RecedingPlan:
        self._require_fresh(observation)
        return SO101RecedingPlan(
            observation=observation,
            target_kind=target_kind,
            target_xyz=target_xyz,
            candidate_name=candidate_name,
            diagnostic_prefix=diagnostic_prefix,
            commands_per_round=self.commands_per_round,
        )

    def authorize(
        self,
        plan: SO101RecedingPlan,
        *,
        current_observation: SO101ObservationIdentity,
    ) -> np.ndarray:
        """Consume the observation and return exactly one executable chunk."""

        if plan.commands_per_round != self.commands_per_round:
            raise ValueError("plan and gate command-chunk contracts do not match")
        if plan.observation != current_observation:
            raise FreshObservationRequired(
                "the selected plan is stale relative to the current paired observation"
            )
        self._require_fresh(current_observation)
        chunk = plan.diagnostic_prefix[: self.commands_per_round].copy()
        self._last_consumed = current_observation
        return chunk


__all__ = [
    "FreshObservationRequired",
    "SO101ObservationIdentity",
    "SO101RecedingHorizonGate",
    "SO101RecedingPlan",
]
