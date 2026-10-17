"""Auditable within-episode trajectories for fixed MuJoCo cameras.

This module deliberately lives outside fcpg.mujoco_camera because the latter
is bound by the frozen PixelObserver G0 protocol. It consumes offsets sampled
by the existing helper and only determines how one endpoint is realized across
recorded, pre-action-aligned frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CAMERA_TRAJECTORY_MODES = ("episode", "frame", "step", "linear_drift")


@dataclass(frozen=True)
class CameraTrajectoryFrame:
    """One realized pose offset and its normalized endpoint phase."""

    translation_world: np.ndarray
    rotation_axis_world: np.ndarray
    rotation_degrees: float
    phase: float

    def as_pose_offset(self) -> tuple[np.ndarray, np.ndarray, float]:
        return (
            np.array(self.translation_world, dtype=np.float64, copy=True),
            np.array(self.rotation_axis_world, dtype=np.float64, copy=True),
            float(self.rotation_degrees),
        )


@dataclass(frozen=True)
class CameraTrajectorySupervision:
    """Per-frame intervention markers for a causal residual prior."""

    intervention_onset: np.ndarray
    prior_supervision_valid: np.ndarray


@dataclass(frozen=True)
class AnchorObservationSchedule:
    """Causal local-anchor dropout and post-dropout recovery annotations."""

    dropout_active: np.ndarray
    recovery_onset: np.ndarray
    recovery_segment: np.ndarray


def anchor_observation_schedule(
    frame_count: int,
    anchor_count: int,
    intervention_onset: np.ndarray,
    *,
    anchor_indices: tuple[int, ...] = (),
    start_offset: int = 1,
    length: int = 0,
    maximum_length: int = 6,
    minimum_recovery_frames: int = 4,
    recovery_length: int | None = None,
) -> AnchorObservationSchedule:
    """Build a local dropout after the first within-episode camera intervention.

    By default a non-zero dropout is limited to four through six frames, which
    preserves the historical diagnostic protocol.  A caller may explicitly
    raise ``maximum_length`` for a separately registered recursive experiment.
    The schedule is an offline dataset annotation and is not exposed as a model
    input.  ``recovery_segment`` begins at the first visible frame after the
    dropout.  It runs to the end of the episode by default; a separately
    registered experiment may provide a finite ``recovery_length``.
    """

    if frame_count <= 0 or anchor_count <= 0:
        raise ValueError("frame_count and anchor_count must be positive")
    onset = np.asarray(intervention_onset)
    if onset.shape != (frame_count,) or onset.dtype != np.bool_:
        raise TypeError("intervention_onset must be a boolean [T] array")
    if start_offset < 0:
        raise ValueError("start_offset must be non-negative")
    if minimum_recovery_frames < 1:
        raise ValueError("minimum_recovery_frames must be positive")
    if recovery_length is not None and recovery_length < 1:
        raise ValueError("recovery_length must be positive when provided")
    if maximum_length < 4:
        raise ValueError("maximum_length must be at least 4")
    if length != 0 and not 4 <= length <= maximum_length:
        raise ValueError(
            "non-zero anchor dropout length must lie in "
            f"[4, {maximum_length}]"
        )
    if len(set(anchor_indices)) != len(anchor_indices):
        raise ValueError("anchor_indices must not contain duplicates")
    if any(index < 0 or index >= anchor_count for index in anchor_indices):
        raise ValueError("anchor index lies outside [0, anchor_count)")

    active = np.zeros((frame_count, anchor_count), dtype=bool)
    recovery_onset = np.zeros_like(active)
    recovery_segment = np.zeros_like(active)
    if length == 0:
        if anchor_indices:
            raise ValueError("anchor_indices require a non-zero dropout length")
        return AnchorObservationSchedule(active, recovery_onset, recovery_segment)
    if not anchor_indices:
        raise ValueError("non-zero anchor dropout requires at least one anchor index")
    events = np.flatnonzero(onset)
    if events.size == 0:
        raise ValueError("anchor dropout requires a camera intervention onset")
    start = int(events[0]) + int(start_offset)
    end = start + int(length)
    if start < 0 or end > frame_count:
        raise ValueError("anchor dropout interval lies outside the recorded episode")
    if frame_count - end < minimum_recovery_frames:
        raise ValueError("anchor dropout leaves too few recovery frames")

    columns = np.asarray(anchor_indices, dtype=np.int64)
    active[start:end, columns] = True
    recovery_onset[end, columns] = True
    recovery_stop = (
        frame_count
        if recovery_length is None
        else min(frame_count, end + int(recovery_length))
    )
    recovery_segment[end:recovery_stop, columns] = True
    return AnchorObservationSchedule(active, recovery_onset, recovery_segment)


def trajectory_supervision_masks(
    mode: str,
    phases: np.ndarray,
    *,
    tolerance: float = 1e-6,
) -> CameraTrajectorySupervision:
    """Return causal supervision masks without exposing schedule time to a model.

    A persistent episode offset has no within-episode surprise. Independently
    randomized frames have no predictable transition. A step masks only its
    discontinuous jump. Linear drift retains its constant-velocity interior and
    masks velocity discontinuities at motion onset or cessation.
    """

    if mode not in CAMERA_TRAJECTORY_MODES:
        raise ValueError(f"unknown camera trajectory mode {mode!r}")
    values = np.asarray(phases, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("phases must be a non-empty one-dimensional array")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("phases must be finite and lie in [0, 1]")
    if not np.isfinite(tolerance) or float(tolerance) < 0.0:
        raise ValueError("tolerance must be finite and non-negative")

    count = int(values.size)
    if mode == "episode":
        return CameraTrajectorySupervision(
            intervention_onset=np.zeros(count, dtype=bool),
            prior_supervision_valid=np.ones(count, dtype=bool),
        )
    if mode == "frame":
        return CameraTrajectorySupervision(
            intervention_onset=np.ones(count, dtype=bool),
            prior_supervision_valid=np.zeros(count, dtype=bool),
        )

    if bool(np.any(np.diff(values) < -float(tolerance))):
        raise ValueError("step and linear-drift phases must be non-decreasing")
    previous = np.concatenate((np.zeros(1, dtype=np.float64), values[:-1]))
    increments = values - previous
    onset = increments > float(tolerance)
    onset[1:] &= increments[:-1] <= float(tolerance)
    if mode == "step":
        valid = ~onset
    else:
        previous_increment = np.concatenate(
            (np.zeros(1, dtype=np.float64), increments[:-1])
        )
        velocity_discontinuity = (
            np.abs(increments - previous_increment) > float(tolerance)
        )
        valid = ~velocity_discontinuity
    return CameraTrajectorySupervision(
        intervention_onset=np.array(onset, dtype=bool, copy=True),
        prior_supervision_valid=np.array(valid, dtype=bool, copy=True),
    )


def trajectory_phase(
    mode: str,
    frame_index: int,
    frame_count: int,
    *,
    step_fraction: float = 0.5,
    drift_start_fraction: float = 0.0,
    drift_end_fraction: float = 1.0,
) -> float:
    """Return trajectory phase for a recorded frame, excluding warm-up frames."""

    if mode not in CAMERA_TRAJECTORY_MODES:
        raise ValueError(
            f"unknown camera trajectory mode {mode!r}; expected one of "
            f"{CAMERA_TRAJECTORY_MODES}"
        )
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if frame_index < 0 or frame_index >= frame_count:
        raise ValueError("frame_index must lie in [0, frame_count)")
    for name, value in (
        ("step_fraction", step_fraction),
        ("drift_start_fraction", drift_start_fraction),
        ("drift_end_fraction", drift_end_fraction),
    ):
        if not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be finite and lie in [0, 1]")
    if float(drift_start_fraction) >= float(drift_end_fraction):
        raise ValueError("drift_start_fraction must be smaller than drift_end_fraction")

    if mode in {"episode", "frame"}:
        return 1.0
    normalized = 1.0 if frame_count == 1 else frame_index / float(frame_count - 1)
    if mode == "step":
        return float(normalized >= float(step_fraction))
    return float(
        np.clip(
            (normalized - float(drift_start_fraction))
            / (float(drift_end_fraction) - float(drift_start_fraction)),
            0.0,
            1.0,
        )
    )


def trajectory_frame(
    mode: str,
    frame_index: int,
    frame_count: int,
    endpoint_offset: tuple[np.ndarray, np.ndarray, float],
    *,
    step_fraction: float = 0.5,
    drift_start_fraction: float = 0.0,
    drift_end_fraction: float = 1.0,
) -> CameraTrajectoryFrame:
    """Interpolate one sampled endpoint without resampling its direction."""

    translation, rotation_axis, rotation_degrees = endpoint_offset
    translation = np.asarray(translation, dtype=np.float64)
    rotation_axis = np.asarray(rotation_axis, dtype=np.float64)
    if translation.shape != (3,) or rotation_axis.shape != (3,):
        raise ValueError("camera endpoint translation and rotation axis need shape [3]")
    if (
        not np.isfinite(translation).all()
        or not np.isfinite(rotation_axis).all()
        or not np.isfinite(rotation_degrees)
    ):
        raise ValueError("camera endpoint must contain only finite values")
    if abs(float(rotation_degrees)) > 1e-12 and np.linalg.norm(rotation_axis) <= 1e-12:
        raise ValueError("non-zero endpoint rotation requires a non-zero axis")

    phase = trajectory_phase(
        mode,
        frame_index,
        frame_count,
        step_fraction=step_fraction,
        drift_start_fraction=drift_start_fraction,
        drift_end_fraction=drift_end_fraction,
    )
    return CameraTrajectoryFrame(
        translation_world=np.array(translation * phase, dtype=np.float64, copy=True),
        rotation_axis_world=np.array(rotation_axis, dtype=np.float64, copy=True),
        rotation_degrees=float(rotation_degrees) * phase,
        phase=phase,
    )
