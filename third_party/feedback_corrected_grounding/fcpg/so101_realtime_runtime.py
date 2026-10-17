"""Runtime contracts for human-confirmed SO-101 receding-horizon control."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any, Callable, Protocol

import numpy as np

from fcpg.so101_realtime_mpc import ManipulationPhase, validate_command_chunk
from fcpg.so101_realtime_planner import (
    CausalPlanningMemory,
    RealtimePlan,
    plan_json,
)
from fcpg.so101_shadow_session import SO101ShadowSession


VIEWS = ("wrist", "column")


@dataclass(frozen=True)
class LiveObservation:
    timestamp_s: float
    state: np.ndarray
    frames: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=np.float32)
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            raise ValueError("live state must be a finite six-vector")
        if not np.isfinite(self.timestamp_s):
            raise ValueError("live timestamp must be finite")
        if set(self.frames) != set(VIEWS):
            raise ValueError("live observation must contain wrist and column frames")
        for view in VIEWS:
            frame = np.asarray(self.frames[view])
            if frame.shape != (224, 224, 3) or frame.dtype != np.uint8:
                raise ValueError(f"{view} frame must be uint8 [224,224,3]")


@dataclass(frozen=True)
class ChunkExecution:
    requested: np.ndarray
    sent: np.ndarray
    observations: tuple[LiveObservation, ...]
    duration_s: float

    def __post_init__(self) -> None:
        requested = np.asarray(self.requested, dtype=np.float32)
        sent = np.asarray(self.sent, dtype=np.float32)
        if requested.shape != (5, 6) or sent.shape != requested.shape:
            raise ValueError("chunk execution commands must have shape [5,6]")
        if not np.all(np.isfinite(requested)) or not np.all(np.isfinite(sent)):
            raise ValueError("chunk execution commands must be finite")
        if len(self.observations) != 5:
            raise ValueError("chunk execution must return five post-command observations")
        if not np.isfinite(self.duration_s) or self.duration_s < 0:
            raise ValueError("chunk duration must be finite and non-negative")


class RealtimeObservationIO(Protocol):
    is_hardware: bool

    def capture_initial(self, count: int, period_s: float) -> list[LiveObservation]: ...

    def observe(self) -> LiveObservation: ...

    def preview_frame(self, view: str) -> np.ndarray: ...

    def execute_chunk(
        self,
        chunk: np.ndarray,
        *,
        period_s: float,
        stop_event: threading.Event,
    ) -> ChunkExecution: ...

    def calibrate_gripper(
        self,
        *,
        stop_event: threading.Event,
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...


class RealtimePlanner(Protocol):
    phase_tracker: Any

    def plan(
        self,
        frames: dict[str, np.ndarray],
        states: np.ndarray,
        *,
        past_raw_actions: np.ndarray | None = None,
        causal_memory: CausalPlanningMemory | None = None,
    ) -> RealtimePlan: ...

    def detect_live_target(
        self,
        column_frame: np.ndarray,
        phase: ManipulationPhase,
        *,
        update_track: bool = False,
    ) -> tuple[str, dict[str, Any]]: ...

    def set_target_reference(
        self,
        role: str,
        view: str,
        frame: np.ndarray,
        centroid_px: np.ndarray,
        *,
        radius_px: float,
    ) -> dict[str, Any]: ...

    def set_live_anchor_center(
        self,
        view: str,
        center_px: np.ndarray,
        state: np.ndarray,
        *,
        anchor_index: int = 0,
    ) -> np.ndarray: ...

    def set_live_anchor_candidates(
        self,
        view: str,
        candidates_px: np.ndarray,
        state: np.ndarray,
        *,
        anchor_index: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def set_live_moving_tip_track(
        self,
        view: str,
        moving_tip_track_candidates_px: np.ndarray,
        open_state: np.ndarray,
        closed_state: np.ndarray,
        *,
        center_anchor_index: int = 0,
        fixed_tip_anchor_index: int = 1,
        moving_tip_anchor_index: int = 2,
    ) -> dict[str, Any]: ...

    def commit_executed_plan(
        self,
        plan: RealtimePlan,
        *,
        command_count: int,
    ) -> dict[str, Any] | None: ...

    def reset_task_progress(self) -> None: ...


ArtifactWriter = Callable[[RealtimePlan], dict[str, str]]


class PlanRejectedError(RuntimeError):
    """A pre-motion safety rejection that permits a fresh observation and replan."""


def _observation_at(session: dict[str, SO101ShadowSession], index: int) -> LiveObservation:
    first = session[VIEWS[0]]
    return LiveObservation(
        timestamp_s=float(first.timestamps_s[index]),
        state=np.asarray(first.states[index], dtype=np.float32).copy(),
        frames={
            view: np.asarray(session[view].frames[index], dtype=np.uint8).copy()
            for view in VIEWS
        },
    )


class ReplaySO101IO:
    """Deterministic paired-session I/O for a no-hardware web dry-run."""

    is_hardware = False

    def __init__(
        self,
        sessions: dict[str, SO101ShadowSession],
        *,
        start_index: int = 10,
    ) -> None:
        if set(sessions) != set(VIEWS):
            raise ValueError("replay requires wrist and column sessions")
        count = sessions[VIEWS[0]].frame_count
        if count < 16:
            raise ValueError("replay requires at least 16 paired frames")
        for view in VIEWS[1:]:
            if sessions[view].frame_count != count:
                raise ValueError("paired replay frame counts differ")
        self.sessions = sessions
        self.count = count
        self.index = max(10, int(start_index))
        if self.index >= count:
            raise ValueError("replay start index is outside the session")
        if self.index + 5 >= count:
            raise ValueError(
                "replay start index must leave a complete next five-frame chunk"
            )

    def capture_initial(self, count: int, period_s: float) -> list[LiveObservation]:
        del period_s
        if count != 11 or self.index < count - 1:
            raise ValueError("replay initial history requires exactly 11 frames")
        return [
            _observation_at(self.sessions, index)
            for index in range(self.index - count + 1, self.index + 1)
        ]

    def observe(self) -> LiveObservation:
        return _observation_at(self.sessions, self.index)

    def preview_frame(self, view: str) -> np.ndarray:
        if view not in VIEWS:
            raise ValueError(f"unsupported preview view: {view}")
        return np.asarray(
            self.sessions[view].frames[self.index], dtype=np.uint8
        ).copy()

    def can_execute_chunk(self) -> bool:
        return self.index + 5 < self.count

    def execute_chunk(
        self,
        chunk: np.ndarray,
        *,
        period_s: float,
        stop_event: threading.Event,
    ) -> ChunkExecution:
        del period_s
        requested = np.asarray(chunk, dtype=np.float32)
        if requested.shape != (5, 6):
            raise ValueError("replay chunk must have shape [5,6]")
        if not self.can_execute_chunk():
            raise RuntimeError("replay session has no complete next five-frame chunk")
        observations: list[LiveObservation] = []
        started = time.perf_counter()
        for _ in range(5):
            if stop_event.is_set():
                raise RuntimeError("execution stopped by operator")
            self.index += 1
            observations.append(_observation_at(self.sessions, self.index))
        return ChunkExecution(
            requested=requested,
            sent=requested.copy(),
            observations=tuple(observations),
            duration_s=time.perf_counter() - started,
        )

    def close(self) -> None:
        return None


class RealtimeMPCController:
    """Coordinate fresh sensing, planning, and bounded closed-loop execution."""

    MAX_CLOSED_LOOP_CHUNKS = 24

    def __init__(
        self,
        planner: RealtimePlanner,
        io: RealtimeObservationIO,
        *,
        artifact_writer: ArtifactWriter | None = None,
        execute_enabled: bool = False,
        calibration_enabled: bool | None = None,
        control_hz: float = 30.0,
        max_plan_age_s: float = 15.0,
        max_state_drift: float = 2.0,
        max_target_drift_px: float = 15.0,
        max_command_jump: float = 8.0,
        minimum_expected_motion: float = 0.5,
        minimum_observed_motion: float = 0.1,
        home_state: np.ndarray | None = None,
        home_control_hz: float = 10.0,
        home_max_step: float = 3.0,
    ) -> None:
        positive = (
            control_hz,
            max_state_drift,
            max_target_drift_px,
            max_command_jump,
            minimum_expected_motion,
            minimum_observed_motion,
            home_control_hz,
            home_max_step,
        )
        if any(not np.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("runtime thresholds must be finite and positive")
        if not np.isfinite(max_plan_age_s) or max_plan_age_s < 0:
            raise ValueError("max_plan_age_s must be finite and non-negative")
        self.planner = planner
        self.io = io
        self.artifact_writer = artifact_writer
        self.execute_enabled = bool(execute_enabled)
        self.calibration_enabled = (
            self.execute_enabled
            if calibration_enabled is None
            else bool(calibration_enabled)
        )
        self.period_s = 1.0 / float(control_hz)
        self.max_plan_age_s = float(max_plan_age_s)
        self.max_state_drift = float(max_state_drift)
        self.max_target_drift_px = float(max_target_drift_px)
        self.max_command_jump = float(max_command_jump)
        self.minimum_expected_motion = float(minimum_expected_motion)
        self.minimum_observed_motion = float(minimum_observed_motion)
        if home_state is None:
            self.home_state = None
        else:
            home = np.asarray(home_state, dtype=np.float32)
            if home.shape != (6,) or not np.all(np.isfinite(home)):
                raise ValueError("home state must be a finite six-vector")
            self.home_state = home.copy()
        self.home_period_s = 1.0 / float(home_control_hz)
        self.home_max_step = float(home_max_step)

        self.observations: deque[LiveObservation] = deque(maxlen=71)
        self.past_chunks: deque[np.ndarray] = deque(maxlen=14)
        self.plan: RealtimePlan | None = None
        self.assets: dict[str, str] = {}
        self.receipts: list[dict[str, Any]] = []
        self.stopped = False
        self.executing = False
        self.active_operation: str | None = None
        self.error: str | None = None
        self.calibration: dict[str, Any] | None = None
        self.last_execution_batch: dict[str, Any] | None = None
        self.execution_armed = False
        self.stop_event = threading.Event()
        self.operation_lock = threading.Lock()
        self.state_lock = threading.Lock()

    def _acquire_operation(self, name: str) -> None:
        if not self.operation_lock.acquire(blocking=False):
            raise RuntimeError("another planning or execution request is active")
        with self.state_lock:
            self.active_operation = name

    def _release_operation(self) -> None:
        with self.state_lock:
            self.active_operation = None
        self.operation_lock.release()

    def bootstrap(self) -> dict[str, Any]:
        with self.operation_lock:
            reset_progress = getattr(self.planner, "reset_task_progress", None)
            if callable(reset_progress):
                reset_progress()
            initial = self.io.capture_initial(11, self.period_s)
            if len(initial) != 11:
                raise RuntimeError("I/O did not return eleven initial observations")
            self.observations.extend(initial)
            self._replan_locked()
            self.execution_armed = True
        return self.status()

    def _history(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        if len(self.observations) < 11:
            raise RuntimeError("runtime does not have a complete 11-frame history")
        selected = [self.observations[index] for index in (-11, -6, -1)]
        frames = {
            view: np.stack([row.frames[view] for row in selected]) for view in VIEWS
        }
        states = np.stack([row.state for row in selected]).astype(np.float32)
        return frames, states

    def _causal_memory(self) -> CausalPlanningMemory | None:
        offsets = tuple(
            int(value) for value in getattr(self.planner, "memory_offsets_raw", ())
        )
        if not offsets:
            return None
        selected_observations: list[LiveObservation] = []
        selected_actions: list[np.ndarray] = []
        selected_offsets: list[int] = []
        for offset in offsets:
            if offset <= 0 or offset % 5:
                raise RuntimeError("planner memory offsets must be positive multiples of 5")
            required_chunks = 2 + offset // 5
            required_observations = 11 + offset
            if (
                len(self.past_chunks) < required_chunks
                or len(self.observations) < required_observations
            ):
                continue
            selected_observations.append(self.observations[-required_observations])
            selected_actions.append(self.past_chunks[-required_chunks])
            selected_offsets.append(offset)
        if not selected_observations:
            return None
        return CausalPlanningMemory(
            frames={
                view: np.stack([row.frames[view] for row in selected_observations])
                for view in VIEWS
            },
            states=np.stack([row.state for row in selected_observations]).astype(
                np.float32
            ),
            raw_actions=np.stack(selected_actions).astype(np.float32),
            offsets_raw=np.asarray(selected_offsets, dtype=np.int64),
        )

    def _replan_locked(self, *, write_artifacts: bool = True) -> None:
        if self.stopped:
            raise RuntimeError("controller is stopped")
        frames, states = self._history()
        recent_chunks = tuple(self.past_chunks)[-2:]
        past = np.stack(recent_chunks) if recent_chunks else None
        plan = self.planner.plan(
            frames,
            states,
            past_raw_actions=past,
            causal_memory=self._causal_memory(),
        )
        assets = (
            self.artifact_writer(plan)
            if write_artifacts and self.artifact_writer
            else None
        )
        with self.state_lock:
            if self.stopped or self.stop_event.is_set():
                raise RuntimeError("controller stopped while planning")
            self.plan = plan
            if assets is not None:
                self.assets = dict(assets)
            self.error = None

    def replan(self) -> dict[str, Any]:
        self._acquire_operation("replan")
        try:
            live = self.io.observe()
            if self.observations:
                self.observations[-1] = live
            else:
                self.observations.append(live)
            self._replan_locked()
            with self.state_lock:
                self.execution_armed = True
            return self.status()
        except BaseException as exc:
            self._latch_fault(exc)
            raise
        finally:
            self._release_operation()

    def set_target_reference(
        self,
        role: str,
        view: str,
        centroid_px: np.ndarray,
        *,
        radius_px: float,
    ) -> dict[str, Any]:
        """Register appearance from the frozen planning frame, then replan."""

        if role not in {"pickup", "drop_zone"}:
            raise ValueError("target role must be pickup or drop_zone")
        if view not in VIEWS:
            raise ValueError("target reference view must be wrist or column")
        if role == "drop_zone" and view != "column":
            raise ValueError("drop_zone must be configured in the column view")
        point = np.asarray(centroid_px, dtype=np.float32)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("target centroid must be a finite two-vector")
        if np.any(point < 0) or np.any(point >= 224):
            raise ValueError("target centroid lies outside the planning frame")
        if not np.isfinite(radius_px) or not 6 <= radius_px <= 112:
            raise ValueError("target radius must be between 6 and 112 pixels")

        self._acquire_operation("set_target_reference")
        try:
            with self.state_lock:
                if self.stopped:
                    raise RuntimeError("controller is stopped")
                plan = self.plan
                if plan is None:
                    raise RuntimeError("no frozen planning frame is available")
                frame = np.asarray(plan.frames[view][-1], dtype=np.uint8).copy()
            self.planner.set_target_reference(
                role,
                view,
                frame,
                point,
                radius_px=float(radius_px),
            )
            reset_progress = getattr(self.planner, "reset_task_progress", None)
            if callable(reset_progress):
                reset_progress()
            self._replan_locked()
            with self.state_lock:
                self.execution_armed = True
            return self.status()
        except ValueError:
            raise
        except BaseException as exc:
            self._latch_fault(exc)
            raise
        finally:
            self._release_operation()

    def calibrate_grounding(self, *, confirmed: bool) -> dict[str, Any]:
        """Run one gripper-only pulse and re-anchor the external column view."""

        if not confirmed:
            raise RuntimeError("grounding calibration requires explicit human confirmation")
        if not self.calibration_enabled:
            raise RuntimeError("gripper calibration is disabled for this server")
        if not self.io.is_hardware:
            raise RuntimeError("grounding calibration is only available on real hardware")
        self._acquire_operation("calibrate_grounding")
        try:
            with self.state_lock:
                if self.stopped:
                    raise RuntimeError("controller is stopped")
                plan = self.plan
                executed = any(
                    receipt.get("kind", "execute_chunk") == "execute_chunk"
                    for receipt in self.receipts
                )
                if plan is None or plan.phase not in {
                    ManipulationPhase.APPROACH,
                    ManipulationPhase.TRAJECTORY,
                }:
                    raise PlanRejectedError(
                        "grounding calibration is only allowed before task execution"
                    )
                if executed:
                    raise PlanRejectedError(
                        "grounding calibration must run before the first task command"
                    )
                self.executing = True
            self.stop_event.clear()
            result = self.io.calibrate_gripper(stop_event=self.stop_event)
            views = result.get("views")
            if not isinstance(views, dict) or set(views) != {"column"}:
                raise RuntimeError("gripper calibration did not return the column view")
            geometry: dict[str, dict[str, Any]] = {}
            for view in ("column",):
                row = views[view]
                moving_tip_tracks = np.asarray(
                    row["moving_tip_track_candidates_px"], dtype=np.float32
                )
                open_state = np.asarray(row["open_state"], dtype=np.float32)
                closed_state = np.asarray(row["closed_state"], dtype=np.float32)
                geometry[view] = self.planner.set_live_moving_tip_track(
                    view,
                    moving_tip_tracks,
                    open_state,
                    closed_state,
                )
            result = {
                **result,
                "observed_anchor": "moving_tip_open_closed_track",
                "geometry": geometry,
            }
            refreshed = self.io.capture_initial(11, self.period_s)
            if len(refreshed) != 11:
                raise RuntimeError(
                    "calibration refresh did not return eleven observations"
                )
            self.observations.clear()
            self.observations.extend(refreshed)
            self.past_chunks.clear()
            self._replan_locked()
            with self.state_lock:
                self.execution_armed = True
                self.calibration = _jsonable(result)
                self.executing = False
            return self.status()
        except PlanRejectedError:
            with self.state_lock:
                self.executing = False
            raise
        except BaseException as exc:
            self._latch_fault(exc)
            raise
        finally:
            self._release_operation()

    def _latch_fault(self, exc: BaseException) -> None:
        self.stop_event.set()
        with self.state_lock:
            if self.stopped and self.error is not None:
                self.executing = False
                return
            self.stopped = True
            self.executing = False
            self.error = f"{type(exc).__name__}: {exc}"
        fault = getattr(self.planner, "phase_tracker", None)
        if fault is not None and hasattr(fault, "fault"):
            fault.fault()

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        with self.state_lock:
            self.stopped = True
            self.executing = False
            if self.error is None:
                self.error = "operator stop latched"
        fault = getattr(self.planner, "phase_tracker", None)
        if fault is not None and hasattr(fault, "fault"):
            fault.fault()
        return self.status()

    def preview_frame(self, view: str) -> np.ndarray:
        """Return a monitor frame without advancing the LeWM observation clock."""

        if view not in VIEWS:
            raise ValueError(f"unsupported preview view: {view}")
        getter = getattr(self.io, "preview_frame", None)
        if not callable(getter):
            raise RuntimeError("runtime I/O does not expose camera previews")
        frame = np.asarray(getter(view))
        if frame.shape != (224, 224, 3) or frame.dtype != np.uint8:
            raise RuntimeError(f"{view} preview must be uint8 [224,224,3]")
        return frame.copy()

    def return_home(self, *, confirmed: bool) -> dict[str, Any]:
        if not confirmed:
            raise RuntimeError("return home requires explicit human confirmation")
        if not self.execute_enabled:
            raise RuntimeError("command execution is disabled for this server")
        if self.home_state is None:
            raise RuntimeError("no task-start home state is configured")
        if not self.io.is_hardware:
            raise RuntimeError("return home is only available on real hardware")
        self._acquire_operation("return_home")

        started = time.perf_counter()
        try:
            with self.state_lock:
                if self.stopped:
                    raise RuntimeError("controller is stopped")
                self.executing = True
            self.stop_event.clear()

            live = self.io.observe()
            maximum_delta = float(np.max(np.abs(self.home_state - live.state)))
            minimum_steps = max(1, int(np.ceil(maximum_delta / self.home_max_step)))
            command_count = 5 * int(np.ceil(minimum_steps / 5))
            path = np.linspace(
                live.state,
                self.home_state,
                command_count + 1,
                dtype=np.float32,
            )[1:]

            sent_chunks: list[np.ndarray] = []
            final_observation = live
            for offset in range(0, command_count, 5):
                chunk = path[offset : offset + 5]
                # Per-step discipline is checked on the PLANNED path. The
                # live-state check below must not count the chunk-boundary
                # jump against home_max_step: the settle loop lets the robot
                # stop anywhere inside its position band, so the settled
                # state lags the previous planned command and the boundary
                # jump is lag + one step — every multi-chunk return home
                # would fail. That jump stays bounded by max_command_jump.
                previous_planned = (
                    live.state if offset == 0 else path[offset - 1]
                )
                planned = np.concatenate(
                    (previous_planned[None], chunk), axis=0
                )
                planned_step = float(np.max(np.abs(np.diff(planned, axis=0))))
                if planned_step > self.home_max_step + 1e-5:
                    raise RuntimeError(
                        "return-home interpolation exceeds the per-step limit"
                    )
                check = validate_command_chunk(
                    chunk,
                    final_observation.state,
                    max_first_jump=self.max_command_jump,
                    max_step_jump=self.max_command_jump,
                    # The recorded home pose may legitimately sit inside the
                    # joint-limit guard band (task-start postures do), so the
                    # final steps toward it are "outward" by definition.
                    # Commands never go past the recorded pose and the hard
                    # position bounds stay enforced here and at the IO layer.
                    outward_tolerance=self.home_max_step + 1e-5,
                )
                if not check.safe:
                    raise RuntimeError(
                        "return-home interpolation failed safety checks: "
                        + ", ".join(check.reasons)
                    )
                result = self.io.execute_chunk(
                    chunk,
                    period_s=self.home_period_s,
                    stop_event=self.stop_event,
                )
                if not np.allclose(
                    result.sent, result.requested, atol=1e-4, rtol=0.0
                ):
                    raise RuntimeError("LeRobot changed a return-home command")
                sent_chunks.append(result.sent.copy())
                final_observation = result.observations[-1]

            receipt = {
                "kind": "return_home",
                "started_monotonic_s": started,
                "duration_s": time.perf_counter() - started,
                "before_state": live.state,
                "requested_home_state": self.home_state,
                "after_state": final_observation.state,
                "maximum_initial_delta": maximum_delta,
                "command_count": command_count,
                "sent": np.concatenate(sent_chunks, axis=0),
                "hardware": True,
            }
            self.receipts.append(receipt)

            refreshed = self.io.capture_initial(11, self.period_s)
            if len(refreshed) != 11:
                raise RuntimeError("home refresh did not return eleven observations")
            self.observations.clear()
            self.observations.extend(refreshed)
            self.past_chunks.clear()
            tracker = getattr(self.planner, "phase_tracker", None)
            if tracker is not None and hasattr(tracker, "reset"):
                tracker.reset()
            reset_grounding = getattr(
                self.planner, "reset_grounding_posterior", None
            )
            if callable(reset_grounding):
                reset_grounding()
            reset_progress = getattr(self.planner, "reset_task_progress", None)
            if callable(reset_progress):
                reset_progress()
            self._replan_locked()
            with self.state_lock:
                self.execution_armed = True
                self.executing = False
            return self.status()
        except BaseException as exc:
            self._latch_fault(exc)
            raise
        finally:
            self._release_operation()

    def _execute_one_locked(self, *, round_id: str) -> dict[str, Any]:
        """Execute and record one five-command token."""

        can_execute = getattr(self.io, "can_execute_chunk", None)
        if callable(can_execute) and not can_execute():
            raise PlanRejectedError(
                "replay session has no complete next five-frame chunk"
            )
        with self.state_lock:
            plan = self.plan
            if self.stopped:
                raise RuntimeError("controller is stopped")
            if not self.execution_armed:
                raise PlanRejectedError(
                    "next chunk is locked; request a fresh plan before execution"
                )
            if plan is None or not plan.executable or plan.selected_chunk is None:
                raise PlanRejectedError(
                    "current planning round is not executable; request a fresh plan"
                )
            self.executing = True

        live = self.io.observe()
        if plan.phase == ManipulationPhase.TRAJECTORY:
            if self.io.is_hardware:
                raise PlanRejectedError(
                    "TRAJECTORY plans are replay-only demonstration "
                    "continuation and cannot execute on hardware"
                )
            # TRAJECTORY has no live target measurement; skip the target-drift
            # check honestly instead of feeding the plan's own target back in.
            descriptor = None
        else:
            try:
                kind, descriptor = self.planner.detect_live_target(
                    live.frames["column"], plan.phase, update_track=False
                )
            except RuntimeError as exc:
                raise PlanRejectedError(f"{exc}; request a fresh plan") from exc
            if kind != plan.guard.target_kind:
                raise PlanRejectedError(
                    "live target kind changed after planning; request a fresh plan"
                )
        try:
            plan.guard.validate_live(
                round_id=round_id,
                now_monotonic_s=time.monotonic(),
                live_state=live.state,
                live_target_px=(
                    None
                    if descriptor is None
                    else np.asarray(descriptor["centroid_px"], dtype=np.float32)
                ),
                max_age_s=self.max_plan_age_s,
                max_state_drift=self.max_state_drift,
                max_target_drift_px=self.max_target_drift_px,
            )
        except (RuntimeError, ValueError) as exc:
            raise PlanRejectedError(f"{exc}; request a fresh plan") from exc
        check = validate_command_chunk(
            plan.selected_chunk,
            live.state,
            max_first_jump=self.max_command_jump,
            max_step_jump=self.max_command_jump,
            # One STS3215 encoder count (360/4096 ~ 0.088): absorbs read-back
            # quantization between the plan-time state sample and this fresh
            # reading; planner-side validation stays strict at 0.0.
            outward_tolerance=0.1,
        )
        if not check.safe:
            raise PlanRejectedError(
                "live command safety failed: "
                + ", ".join(check.reasons)
                + "; request a fresh plan"
            )

        started = time.perf_counter()
        result = self.io.execute_chunk(
            plan.selected_chunk,
            period_s=self.period_s,
            stop_event=self.stop_event,
        )
        if not np.allclose(result.sent, result.requested, atol=1e-4, rtol=0.0):
            raise RuntimeError("LeRobot clipped or changed a command")
        # Arm joints only: the gripper legitimately stalls against a grasped
        # object without reaching its command, and that stall must not read
        # as "robot did not follow".
        expected_motion = float(
            np.max(np.abs(result.requested[-1][:5] - live.state[:5]))
        )
        observed_motion = float(
            np.max(np.abs(result.observations[-1].state[:5] - live.state[:5]))
        )
        if (
            self.io.is_hardware
            and expected_motion >= self.minimum_expected_motion
            and observed_motion < self.minimum_observed_motion
        ):
            raise RuntimeError(
                "robot did not follow the five-command chunk: "
                f"expected={expected_motion:.3f}, observed={observed_motion:.3f}"
            )

        # Only completed commands advance model time. Every next token is
        # planned from the endpoint observations returned by this execution.
        self.observations.extend(result.observations)
        self.past_chunks.append(result.sent.copy())
        progress: dict[str, Any] | None = None
        commit_progress = getattr(self.planner, "commit_executed_plan", None)
        if callable(commit_progress):
            progress = commit_progress(plan, command_count=len(result.sent))
        receipt = {
            "kind": "execute_chunk",
            "round_id": plan.guard.round_id,
            "phase": plan.phase.value,
            "target_kind": plan.guard.target_kind,
            "started_monotonic_s": started,
            "duration_s": result.duration_s,
            "requested": result.requested,
            "sent": result.sent,
            "before_state": live.state,
            "after_state": result.observations[-1].state,
            "expected_motion": expected_motion,
            "observed_motion": observed_motion,
            "task_progress": progress,
            "hardware": bool(self.io.is_hardware),
            "target_check": (
                "skipped_no_live_target_in_trajectory"
                if descriptor is None
                else "live_detected"
            ),
        }
        self.receipts.append(receipt)
        return receipt

    def execute(
        self,
        *,
        round_id: str,
        confirmed: bool,
        max_chunks: int = 1,
    ) -> dict[str, Any]:
        """Run up to four closed-loop five-command tokens from one request."""

        if not confirmed:
            raise RuntimeError("execution requires explicit human confirmation")
        if not self.execute_enabled:
            raise RuntimeError("command execution is disabled for this server")
        if isinstance(max_chunks, bool) or not isinstance(max_chunks, int):
            raise ValueError("max_chunks must be an integer")
        if not 1 <= max_chunks <= self.MAX_CLOSED_LOOP_CHUNKS:
            raise ValueError(
                f"max_chunks must be between 1 and {self.MAX_CLOSED_LOOP_CHUNKS}"
            )

        operation = "execute_chunk" if max_chunks == 1 else "execute_20_commands"
        self._acquire_operation(operation)
        batch_started = time.perf_counter()
        completed: list[dict[str, Any]] = []
        stop_reason = "command_budget_exhausted"
        try:
            with self.state_lock:
                if self.stopped:
                    raise RuntimeError("controller is stopped")
                self.stop_event.clear()
            current_round_id = round_id
            for index in range(max_chunks):
                if index:
                    can_execute = getattr(self.io, "can_execute_chunk", None)
                    if callable(can_execute) and not can_execute():
                        stop_reason = "input_exhausted"
                        break
                    with self.state_lock:
                        plan = self.plan
                        if (
                            plan is None
                            or not plan.executable
                            or plan.selected_chunk is None
                        ):
                            stop_reason = "next_plan_not_executable"
                            break
                        current_round_id = plan.guard.round_id
                receipt = self._execute_one_locked(round_id=current_round_id)
                completed.append(receipt)
                self._replan_locked(write_artifacts=False)

            with self.state_lock:
                final_plan = self.plan
            if self.artifact_writer is not None and final_plan is not None:
                final_assets = self.artifact_writer(final_plan)
                with self.state_lock:
                    self.assets = dict(final_assets)

            batch = {
                "requested_chunks": max_chunks,
                "requested_commands": max_chunks * 5,
                "executed_chunks": len(completed),
                "executed_commands": len(completed) * 5,
                "stop_reason": stop_reason,
                "round_ids": [row["round_id"] for row in completed],
                "phases": [row["phase"] for row in completed],
                "duration_s": time.perf_counter() - batch_started,
            }
            with self.state_lock:
                self.last_execution_batch = batch
                self.execution_armed = False
                self.executing = False
            return self.status()
        except PlanRejectedError:
            with self.state_lock:
                self.execution_armed = False
                self.executing = False
                self.last_execution_batch = {
                    "requested_chunks": max_chunks,
                    "requested_commands": max_chunks * 5,
                    "executed_chunks": len(completed),
                    "executed_commands": len(completed) * 5,
                    "stop_reason": "plan_rejected",
                    "round_ids": [row["round_id"] for row in completed],
                    "phases": [row["phase"] for row in completed],
                    "duration_s": time.perf_counter() - batch_started,
                }
            raise
        except BaseException as exc:
            with self.state_lock:
                self.last_execution_batch = {
                    "requested_chunks": max_chunks,
                    "requested_commands": max_chunks * 5,
                    "executed_chunks": len(completed),
                    "executed_commands": len(completed) * 5,
                    "stop_reason": "runtime_fault",
                    "round_ids": [row["round_id"] for row in completed],
                    "phases": [row["phase"] for row in completed],
                    "duration_s": time.perf_counter() - batch_started,
                }
            self._latch_fault(exc)
            raise
        finally:
            self._release_operation()

    def status(self) -> dict[str, Any]:
        with self.state_lock:
            plan = self.plan
            can_execute = getattr(self.io, "can_execute_chunk", None)
            io_ready = not callable(can_execute) or bool(can_execute())
            payload = {
                "mode": "hardware" if self.io.is_hardware else "replay",
                "execute_enabled": self.execute_enabled and io_ready,
                "calibration_enabled": self.calibration_enabled,
                "home_available": bool(
                    self.io.is_hardware and self.home_state is not None
                ),
                "input_exhausted": not io_ready,
                "stopped": self.stopped,
                "executing": self.executing,
                "execution_armed": self.execution_armed,
                "active_operation": self.active_operation,
                "error": self.error,
                "calibration_available": bool(
                    self.io.is_hardware
                    and self.calibration_enabled
                    and not self.stopped
                    and self.active_operation is None
                    and plan is not None
                    and plan.phase in {
                        ManipulationPhase.APPROACH,
                        ManipulationPhase.TRAJECTORY,
                    }
                    and not any(
                        receipt.get("kind", "execute_chunk") == "execute_chunk"
                        for receipt in self.receipts
                    )
                ),
                "calibration": self.calibration,
                "observation_clock": "task_event",
                "monitor_frames_feed_model": False,
                "model_history_frames": len(self.observations),
                "model_executed_chunks": len(self.past_chunks),
                "model_memory_tokens": (
                    0
                    if (memory := self._causal_memory()) is None
                    else len(memory.offsets_raw)
                ),
                "assets": dict(self.assets),
                "completed_chunks": sum(
                    receipt.get("kind", "execute_chunk") == "execute_chunk"
                    for receipt in self.receipts
                ),
                "latest_receipt": self.receipts[-1] if self.receipts else None,
                "last_execution_batch": self.last_execution_batch,
                "plan": None if plan is None else plan_json(plan),
            }
        return _jsonable(payload)

    def close(self) -> None:
        self.stop_event.set()
        self.io.close()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "ChunkExecution",
    "LiveObservation",
    "PlanRejectedError",
    "RealtimeMPCController",
    "RealtimeObservationIO",
    "ReplaySO101IO",
]
