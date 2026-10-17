import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from fcpg.so101_realtime_mpc import ManipulationPhase, PlanningRoundGuard
from fcpg.so101_realtime_runtime import (
    ChunkExecution,
    LiveObservation,
    PlanRejectedError,
    RealtimeMPCController,
    ReplaySO101IO,
)


def _observation(state: np.ndarray | None = None, value: int = 0) -> LiveObservation:
    return LiveObservation(
        timestamp_s=float(value),
        state=np.zeros(6, dtype=np.float32) if state is None else state,
        frames={
            "wrist": np.full((224, 224, 3), value, dtype=np.uint8),
            "column": np.full((224, 224, 3), value, dtype=np.uint8),
        },
    )


class FakePhaseTracker:
    def __init__(self) -> None:
        self.faulted = False
        self.reset_count = 0

    def fault(self) -> None:
        self.faulted = True

    def reset(self) -> None:
        self.reset_count += 1
        self.faulted = False


class FakePlan:
    def __init__(
        self,
        state: np.ndarray,
        chunk: np.ndarray,
        frames: dict[str, np.ndarray],
    ) -> None:
        self.guard = PlanningRoundGuard(
            round_id=f"round-{time.time_ns()}",
            created_monotonic_s=time.monotonic(),
            state=np.asarray(state, dtype=np.float32),
            target_px=np.asarray([20.0, 30.0], dtype=np.float32),
            target_kind="red_object",
        )
        self.phase = ManipulationPhase.APPROACH
        self.executable = True
        self.selected_chunk = np.asarray(chunk, dtype=np.float32)
        self.frames = {
            view: np.asarray(values, dtype=np.uint8).copy()
            for view, values in frames.items()
        }

    def summary(self) -> dict[str, object]:
        return {
            "round_id": self.guard.round_id,
            "phase": self.phase.value,
            "executable": self.executable,
            "selected_chunk": self.selected_chunk,
        }


class FakePlanner:
    def __init__(self, chunk: np.ndarray) -> None:
        self.chunk = np.asarray(chunk, dtype=np.float32)
        self.memory_offsets_raw = (5, 15, 30, 60)
        self.phase_tracker = FakePhaseTracker()
        self.histories: list[np.ndarray] = []
        self.memories: list[object | None] = []
        self.target_references: list[dict[str, object]] = []
        self.live_anchor_centers: dict[
            str, tuple[np.ndarray, np.ndarray, int]
        ] = {}
        self.live_anchor_geometry: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.grounding_reset_count = 0
        self.progress_reset_count = 0
        self.committed_plans: list[tuple[FakePlan, int]] = []

    def reset_grounding_posterior(self) -> None:
        self.grounding_reset_count += 1

    def reset_task_progress(self) -> None:
        self.progress_reset_count += 1

    def commit_executed_plan(
        self,
        plan: FakePlan,
        *,
        command_count: int,
    ) -> dict[str, int]:
        self.committed_plans.append((plan, command_count))
        return {"completed_chunks": len(self.committed_plans)}

    def set_target_reference(
        self,
        role: str,
        view: str,
        frame: np.ndarray,
        centroid_px: np.ndarray,
        *,
        radius_px: float,
    ) -> dict[str, object]:
        record: dict[str, object] = {
            "role": role,
            "view": view,
            "frame": np.asarray(frame, dtype=np.uint8).copy(),
            "centroid_px": np.asarray(centroid_px, dtype=np.float32).copy(),
            "radius_px": float(radius_px),
        }
        self.target_references.append(record)
        return record

    def set_live_anchor_center(
        self,
        view: str,
        center_px: np.ndarray,
        state: np.ndarray,
        *,
        anchor_index: int = 0,
    ) -> np.ndarray:
        center = np.asarray(center_px, dtype=np.float32).copy()
        calibration_state = np.asarray(state, dtype=np.float32).copy()
        self.live_anchor_centers[view] = (
            center,
            calibration_state,
            int(anchor_index),
        )
        return np.asarray([3.0, -4.0], dtype=np.float32)

    def set_live_anchor_candidates(
        self,
        view: str,
        candidates_px: np.ndarray,
        state: np.ndarray,
        *,
        anchor_index: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        candidates = np.asarray(candidates_px, dtype=np.float32)
        selected = candidates[-1].copy()
        calibration_state = np.asarray(state, dtype=np.float32).copy()
        self.live_anchor_centers[view] = (
            selected,
            calibration_state,
            int(anchor_index),
        )
        return selected, np.asarray([3.0, -4.0], dtype=np.float32)

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
    ) -> dict[str, object]:
        del center_anchor_index, fixed_tip_anchor_index, moving_tip_anchor_index
        candidates = np.asarray(
            moving_tip_track_candidates_px, dtype=np.float32
        ).copy()
        open_calibration_state = np.asarray(open_state, dtype=np.float32).copy()
        closed_calibration_state = np.asarray(closed_state, dtype=np.float32).copy()
        self.live_anchor_geometry[view] = (
            candidates,
            open_calibration_state,
            closed_calibration_state,
        )
        return {
            "candidate_index": 1,
            "observed_moving_tip_track_px": candidates[1],
            "transform_px": np.asarray(
                [[0.75, 0.0, 8.0], [0.0, 0.75, -4.0]], dtype=np.float32
            ),
            "scale": 0.75,
            "rotation_deg": 0.0,
        }

    def plan(
        self,
        frames: dict[str, np.ndarray],
        states: np.ndarray,
        *,
        past_raw_actions: np.ndarray | None = None,
        causal_memory=None,
    ) -> FakePlan:
        assert frames["wrist"].shape == (3, 224, 224, 3)
        if past_raw_actions is not None:
            assert past_raw_actions.shape[-2:] == (5, 6)
        self.histories.append(np.asarray(states).copy())
        self.memories.append(causal_memory)
        return FakePlan(states[-1], self.chunk, frames)

    def detect_live_target(
        self,
        column_frame: np.ndarray,
        phase: ManipulationPhase,
        *,
        update_track: bool = False,
    ) -> tuple[str, dict[str, object]]:
        del column_frame, phase, update_track
        return "red_object", {
            "visible": True,
            "centroid_px": np.asarray([20.0, 30.0], dtype=np.float32),
        }


class StopAfterOnePlanner(FakePlanner):
    def plan(
        self,
        frames: dict[str, np.ndarray],
        states: np.ndarray,
        *,
        past_raw_actions: np.ndarray | None = None,
        causal_memory=None,
    ) -> FakePlan:
        plan = super().plan(
            frames,
            states,
            past_raw_actions=past_raw_actions,
            causal_memory=causal_memory,
        )
        if len(self.histories) >= 2:
            plan.executable = False
            plan.selected_chunk = None
        return plan


class FailAfterOnePlanner(FakePlanner):
    def plan(
        self,
        frames: dict[str, np.ndarray],
        states: np.ndarray,
        *,
        past_raw_actions: np.ndarray | None = None,
        causal_memory=None,
    ) -> FakePlan:
        plan = super().plan(
            frames,
            states,
            past_raw_actions=past_raw_actions,
            causal_memory=causal_memory,
        )
        if len(self.histories) >= 2:
            raise RuntimeError("synthetic replanning failure")
        return plan


class FakeIO:
    is_hardware = True

    def __init__(self, *, changed_command: bool = False, no_motion: bool = False) -> None:
        self.current = _observation(value=10)
        self.changed_command = changed_command
        self.no_motion = no_motion
        self.executed: list[np.ndarray] = []
        self.capture_counts: list[int] = []
        self.calibration_calls = 0
        self.preview_calls: list[str] = []

    def capture_initial(self, count: int, period_s: float) -> list[LiveObservation]:
        del period_s
        self.capture_counts.append(count)
        return [_observation(value=index) for index in range(count)]

    def observe(self) -> LiveObservation:
        return self.current

    def preview_frame(self, view: str) -> np.ndarray:
        self.preview_calls.append(view)
        return self.current.frames[view].copy()

    def execute_chunk(
        self,
        chunk: np.ndarray,
        *,
        period_s: float,
        stop_event: threading.Event,
    ) -> ChunkExecution:
        del period_s
        assert not stop_event.is_set()
        values = np.asarray(chunk, dtype=np.float32)
        self.executed.append(values.copy())
        sent = values.copy()
        if self.changed_command:
            sent[2, 0] += 0.25
        rows = []
        for index in range(5):
            state = (
                np.zeros(6, dtype=np.float32)
                if self.no_motion
                else values[index].copy()
            )
            rows.append(_observation(state=state, value=11 + index))
        self.current = rows[-1]
        return ChunkExecution(
            requested=values,
            sent=sent,
            observations=tuple(rows),
            duration_s=0.1,
        )

    def calibrate_gripper(
        self,
        *,
        stop_event: threading.Event,
    ) -> dict[str, object]:
        assert not stop_event.is_set()
        self.calibration_calls += 1
        state = np.asarray([1, 2, 3, 4, 5, 32], dtype=np.float32)
        return {
            "center_px": np.asarray([72.0, 88.0], dtype=np.float32),
            "state": state,
            "opening_span": 55.0,
            "views": {
                "column": {
                    "open_tip_px": np.asarray([82.0, 92.0], dtype=np.float32),
                    "moving_tip_track_candidates_px": np.asarray(
                        [
                            [[76.0, 28.0], [88.0, 28.0]],
                            [[70.0, 94.0], [98.0, 94.0]],
                        ],
                        dtype=np.float32,
                    ),
                    "open_state": state,
                    "closed_state": np.asarray(
                        [1, 2, 3, 4, 5, 2], dtype=np.float32
                    ),
                },
            },
        }

    def close(self) -> None:
        return None


def _chunk() -> np.ndarray:
    values = np.zeros((5, 6), dtype=np.float32)
    values[:, 0] = np.linspace(0.2, 1.0, 5)
    return values


def _controller(
    io: FakeIO, *, home_state: np.ndarray | None = None
) -> tuple[RealtimeMPCController, FakePlanner]:
    planner = FakePlanner(_chunk())
    controller = RealtimeMPCController(
        planner,
        io,
        execute_enabled=True,
        max_plan_age_s=10.0,
        home_state=home_state,
    )
    controller.bootstrap()
    return controller, planner


def test_bootstrap_selects_t_minus_10_t_minus_5_and_t() -> None:
    controller, planner = _controller(FakeIO())
    np.testing.assert_array_equal(planner.histories[0][:, 0], [0.0, 0.0, 0.0])
    values = [int(frame[0, 0, 0]) for frame in (
        controller.observations[0].frames["wrist"],
        controller.observations[5].frames["wrist"],
        controller.observations[10].frames["wrist"],
    )]
    assert values == [0, 5, 10]


def test_monitor_preview_does_not_advance_model_history() -> None:
    io = FakeIO()
    controller, planner = _controller(io)
    observations_before = tuple(controller.observations)
    history_before = [history.copy() for history in planner.histories]
    status_before = controller.status()

    for _ in range(6):
        wrist = controller.preview_frame("wrist")
        column = controller.preview_frame("column")

    np.testing.assert_array_equal(wrist, io.current.frames["wrist"])
    np.testing.assert_array_equal(column, io.current.frames["column"])
    assert len(controller.observations) == len(observations_before)
    assert all(
        actual is expected
        for actual, expected in zip(
            controller.observations, observations_before, strict=True
        )
    )
    assert len(planner.histories) == len(history_before)
    for actual, expected in zip(planner.histories, history_before, strict=True):
        np.testing.assert_array_equal(actual, expected)
    status_after = controller.status()
    assert status_after["model_history_frames"] == status_before["model_history_frames"]
    assert status_after["completed_chunks"] == status_before["completed_chunks"]
    assert status_after["observation_clock"] == "task_event"
    assert status_after["monitor_frames_feed_model"] is False
    assert io.preview_calls == ["wrist", "column"] * 6


def test_monitor_preview_rejects_unknown_view() -> None:
    controller, _ = _controller(FakeIO())

    with pytest.raises(ValueError, match="unsupported preview view"):
        controller.preview_frame("overhead")


def test_trajectory_plan_is_rejected_on_hardware() -> None:
    io = FakeIO()
    controller, _ = _controller(io)
    controller.plan.phase = ManipulationPhase.TRAJECTORY
    round_id = controller.plan.guard.round_id

    with pytest.raises(PlanRejectedError, match="replay-only"):
        controller.execute(round_id=round_id, confirmed=True)

    assert io.executed == []


def test_trajectory_replay_skips_target_check_honestly() -> None:
    io = FakeIO()
    io.is_hardware = False
    planner = StopAfterOnePlanner(_chunk())

    def _fail_detect(*args: object, **kwargs: object) -> None:
        raise AssertionError("detect_live_target must not run in TRAJECTORY")

    planner.detect_live_target = _fail_detect  # type: ignore[method-assign]
    controller = RealtimeMPCController(
        planner,
        io,
        execute_enabled=True,
        max_plan_age_s=10.0,
    )
    controller.bootstrap()
    controller.plan.phase = ManipulationPhase.TRAJECTORY
    round_id = controller.plan.guard.round_id

    controller.execute(round_id=round_id, confirmed=True)

    assert len(io.executed) == 1
    receipt = next(
        record
        for record in controller.receipts
        if record.get("kind") == "execute_chunk"
    )
    assert receipt["target_check"] == "skipped_no_live_target_in_trajectory"


def test_execute_sends_exactly_five_commands_then_replans() -> None:
    io = FakeIO()
    controller, planner = _controller(io)
    round_id = controller.plan.guard.round_id

    result = controller.execute(round_id=round_id, confirmed=True)

    assert len(io.executed) == 1
    assert io.executed[0].shape == (5, 6)
    assert io.capture_counts == [11]
    assert len(planner.histories) == 2
    assert len(planner.committed_plans) == 1
    assert planner.committed_plans[0][1] == 5
    assert result["latest_receipt"]["task_progress"] == {"completed_chunks": 1}
    assert result["completed_chunks"] == 1
    assert result["execution_armed"] is False
    assert not result["stopped"]


def test_execute_twenty_commands_as_four_fresh_five_command_rounds() -> None:
    io = FakeIO()
    controller, planner = _controller(io)
    first_round = controller.plan.guard.round_id

    result = controller.execute(
        round_id=first_round,
        confirmed=True,
        max_chunks=4,
    )

    assert len(io.executed) == 4
    assert all(chunk.shape == (5, 6) for chunk in io.executed)
    assert len(planner.histories) == 5
    assert result["completed_chunks"] == 4
    assert result["model_executed_chunks"] == 4
    assert result["execution_armed"] is False
    assert result["last_execution_batch"]["requested_commands"] == 20
    assert result["last_execution_batch"]["executed_commands"] == 20
    assert result["last_execution_batch"]["stop_reason"] == "command_budget_exhausted"
    assert len(set(result["last_execution_batch"]["round_ids"])) == 4
    assert not result["stopped"]


def test_execute_twenty_renders_only_initial_and_final_plans() -> None:
    io = FakeIO()
    planner = FakePlanner(_chunk())
    rendered_rounds: list[str] = []
    controller = RealtimeMPCController(
        planner,
        io,
        artifact_writer=lambda plan: (
            rendered_rounds.append(plan.guard.round_id) or {}
        ),
        execute_enabled=True,
        max_plan_age_s=0.0,
    )
    controller.bootstrap()

    controller.execute(
        round_id=controller.plan.guard.round_id,
        confirmed=True,
        max_chunks=4,
    )

    assert len(rendered_rounds) == 2


def test_execute_twenty_stops_when_fresh_plan_is_not_executable() -> None:
    io = FakeIO()
    planner = StopAfterOnePlanner(_chunk())
    controller = RealtimeMPCController(
        planner,
        io,
        execute_enabled=True,
        max_plan_age_s=10.0,
    )
    controller.bootstrap()

    result = controller.execute(
        round_id=controller.plan.guard.round_id,
        confirmed=True,
        max_chunks=4,
    )

    assert len(io.executed) == 1
    assert len(planner.histories) == 2
    assert result["completed_chunks"] == 1
    assert result["last_execution_batch"]["executed_commands"] == 5
    assert (
        result["last_execution_batch"]["stop_reason"]
        == "next_plan_not_executable"
    )
    assert not result["stopped"]


def test_execute_twenty_records_completed_chunk_before_replanning_fault() -> None:
    io = FakeIO()
    planner = FailAfterOnePlanner(_chunk())
    controller = RealtimeMPCController(
        planner,
        io,
        execute_enabled=True,
        max_plan_age_s=10.0,
    )
    controller.bootstrap()

    with pytest.raises(RuntimeError, match="synthetic replanning failure"):
        controller.execute(
            round_id=controller.plan.guard.round_id,
            confirmed=True,
            max_chunks=4,
        )

    result = controller.status()
    assert len(io.executed) == 1
    assert result["completed_chunks"] == 1
    assert result["last_execution_batch"]["executed_commands"] == 5
    assert result["last_execution_batch"]["stop_reason"] == "runtime_fault"
    assert result["stopped"] is True


@pytest.mark.parametrize("max_chunks", [0, 25, True, 1.5])
def test_execute_rejects_invalid_closed_loop_budget_without_motion(max_chunks) -> None:
    io = FakeIO()
    controller, _ = _controller(io)

    with pytest.raises(ValueError, match="max_chunks"):
        controller.execute(
            round_id=controller.plan.guard.round_id,
            confirmed=True,
            max_chunks=max_chunks,
        )

    assert not io.executed


def test_v6_memory_uses_aligned_decision_state_and_following_chunk() -> None:
    io = FakeIO()
    controller, planner = _controller(io)

    first_chunk = controller.plan.selected_chunk.copy()
    for _ in range(3):
        controller.execute(round_id=controller.plan.guard.round_id, confirmed=True)
        if _ < 2:
            controller.replan()

    memory = planner.memories[-1]
    assert memory is not None
    np.testing.assert_array_equal(memory.offsets_raw, [5])
    np.testing.assert_array_equal(memory.states[0], np.zeros(6, dtype=np.float32))
    np.testing.assert_array_equal(memory.raw_actions[0], first_chunk)
    assert int(memory.frames["wrist"][0, 0, 0, 0]) == 10


def test_execute_requires_explicit_replan_before_next_chunk() -> None:
    io = FakeIO()
    controller, _ = _controller(io)
    controller.execute(round_id=controller.plan.guard.round_id, confirmed=True)

    with pytest.raises(RuntimeError, match="request a fresh plan"):
        controller.execute(round_id=controller.plan.guard.round_id, confirmed=True)

    assert len(io.executed) == 1
    assert not controller.stopped
    controller.replan()
    assert controller.status()["execution_armed"] is True
    controller.execute(round_id=controller.plan.guard.round_id, confirmed=True)
    assert len(io.executed) == 2


def test_return_home_uses_slow_five_command_chunks_then_resets_history() -> None:
    io = FakeIO()
    home = np.zeros(6, dtype=np.float32)
    controller, planner = _controller(io, home_state=home)
    io.current = _observation(state=np.full(6, 12.0, dtype=np.float32), value=20)

    result = controller.return_home(confirmed=True)

    assert len(io.executed) == 1
    assert io.executed[0].shape == (5, 6)
    assert np.max(np.abs(np.diff(io.executed[0], axis=0))) <= 3.0
    np.testing.assert_allclose(io.executed[0][-1], home)
    assert io.capture_counts == [11, 11]
    assert planner.phase_tracker.reset_count == 1
    assert planner.grounding_reset_count == 1
    assert planner.progress_reset_count == 2
    assert result["home_available"]
    assert result["latest_receipt"]["kind"] == "return_home"


class GripperLagIO(FakeIO):
    """Settle leaves the gripper 2.0 short of the last command."""

    def execute_chunk(
        self,
        chunk: np.ndarray,
        *,
        period_s: float,
        stop_event: threading.Event,
    ) -> ChunkExecution:
        result = super().execute_chunk(
            chunk, period_s=period_s, stop_event=stop_event
        )
        rows = list(result.observations)
        lagged = rows[-1].state.copy()
        lagged[5] -= 2.0
        rows[-1] = _observation(state=lagged, value=99)
        self.current = rows[-1]
        return ChunkExecution(
            requested=result.requested,
            sent=result.sent,
            observations=tuple(rows),
            duration_s=result.duration_s,
        )


def test_return_home_reaches_a_pose_inside_the_limit_guard_band() -> None:
    io = FakeIO()
    home = np.zeros(6, dtype=np.float32)
    home[1] = -98.0
    controller, _ = _controller(io, home_state=home)

    result = controller.return_home(confirmed=True)

    np.testing.assert_allclose(io.executed[-1][-1], home)
    assert result["latest_receipt"]["kind"] == "return_home"


def test_return_home_tolerates_settle_lag_across_chunks() -> None:
    io = GripperLagIO()
    home = np.zeros(6, dtype=np.float32)
    home[5] = 30.0
    controller, _ = _controller(io, home_state=home)

    result = controller.return_home(confirmed=True)

    assert len(io.executed) == 2
    for chunk in io.executed:
        assert np.max(np.abs(np.diff(chunk, axis=0))) <= 3.0 + 1e-5
    np.testing.assert_allclose(io.executed[-1][-1], home)
    assert result["latest_receipt"]["kind"] == "return_home"


def test_calibration_reanchors_column_without_executing_an_arm_chunk() -> None:
    io = FakeIO()
    controller, planner = _controller(io)
    assert controller.status()["calibration_available"]

    result = controller.calibrate_grounding(confirmed=True)

    assert io.calibration_calls == 1
    assert not io.executed
    assert io.capture_counts == [11, 11]
    assert set(planner.live_anchor_geometry) == {"column"}
    candidates, open_state, closed_state = planner.live_anchor_geometry["column"]
    np.testing.assert_array_equal(
        candidates,
        [
            [[76.0, 28.0], [88.0, 28.0]],
            [[70.0, 94.0], [98.0, 94.0]],
        ],
    )
    np.testing.assert_array_equal(open_state, [1, 2, 3, 4, 5, 32])
    np.testing.assert_array_equal(closed_state, [1, 2, 3, 4, 5, 2])
    assert result["calibration"]["geometry"]["column"]["scale"] == 0.75
    assert result["completed_chunks"] == 0


def test_calibration_permission_does_not_enable_task_execution() -> None:
    io = FakeIO()
    planner = FakePlanner(_chunk())
    controller = RealtimeMPCController(
        planner,
        io,
        execute_enabled=False,
        calibration_enabled=True,
        max_plan_age_s=10.0,
    )
    controller.bootstrap()

    status = controller.status()
    assert status["calibration_enabled"] is True
    assert status["calibration_available"] is True
    assert status["execute_enabled"] is False

    with pytest.raises(RuntimeError, match="command execution is disabled"):
        controller.execute(round_id=controller.plan.guard.round_id, confirmed=True)

    result = controller.calibrate_grounding(confirmed=True)
    assert io.calibration_calls == 1
    assert not io.executed
    assert result["completed_chunks"] == 0


def test_calibration_is_rejected_after_first_task_chunk() -> None:
    io = FakeIO()
    controller, _ = _controller(io)
    controller.execute(round_id=controller.plan.guard.round_id, confirmed=True)

    with pytest.raises(RuntimeError, match="before the first task command"):
        controller.calibrate_grounding(confirmed=True)

    assert io.calibration_calls == 0
    assert not controller.stopped


def test_execute_rejects_unconfirmed_request_without_motion() -> None:
    io = FakeIO()
    controller, _ = _controller(io)
    with pytest.raises(RuntimeError, match="explicit human confirmation"):
        controller.execute(round_id=controller.plan.guard.round_id, confirmed=False)
    assert not io.executed


def test_stale_round_rejects_without_latching_before_motion() -> None:
    io = FakeIO()
    controller, planner = _controller(io)
    with pytest.raises(RuntimeError, match="stale planning round"):
        controller.execute(round_id="old-round", confirmed=True)
    assert not io.executed
    assert not controller.stopped
    assert not planner.phase_tracker.faulted

    controller.replan()
    assert controller.error is None


def test_changed_command_latches_fault() -> None:
    io = FakeIO(changed_command=True)
    controller, planner = _controller(io)
    with pytest.raises(RuntimeError, match="clipped or changed"):
        controller.execute(round_id=controller.plan.guard.round_id, confirmed=True)
    assert controller.stopped
    assert planner.phase_tracker.faulted
    assert not planner.committed_plans


def test_no_observed_motion_latches_fault() -> None:
    io = FakeIO(no_motion=True)
    controller, planner = _controller(io)
    with pytest.raises(RuntimeError, match="did not follow"):
        controller.execute(round_id=controller.plan.guard.round_id, confirmed=True)
    assert controller.stopped
    assert planner.phase_tracker.faulted
    assert not planner.committed_plans


def test_operator_stop_prevents_execution() -> None:
    io = FakeIO()
    controller, planner = _controller(io)
    controller.stop()
    with pytest.raises(RuntimeError, match="controller is stopped"):
        controller.execute(round_id=controller.plan.guard.round_id, confirmed=True)
    assert not io.executed
    assert planner.phase_tracker.faulted


def test_replan_after_latched_fault_preserves_original_error() -> None:
    io = FakeIO(changed_command=True)
    controller, _ = _controller(io)
    with pytest.raises(RuntimeError, match="clipped or changed"):
        controller.execute(round_id=controller.plan.guard.round_id, confirmed=True)
    original = controller.error

    with pytest.raises(RuntimeError, match="controller is stopped"):
        controller.replan()

    assert controller.error == original


def test_target_reference_uses_frozen_plan_frame_and_replans_without_motion() -> None:
    io = FakeIO()
    controller, planner = _controller(io)
    controller.set_target_reference(
        "pickup",
        "wrist",
        np.asarray([110.0, 70.0]),
        radius_px=18.0,
    )

    reference = planner.target_references[-1]
    assert reference["role"] == "pickup"
    assert reference["view"] == "wrist"
    assert reference["radius_px"] == 18.0
    np.testing.assert_array_equal(reference["centroid_px"], [110.0, 70.0])
    np.testing.assert_array_equal(
        reference["frame"], np.full((224, 224, 3), 10, dtype=np.uint8)
    )
    assert len(planner.histories) == 2
    assert planner.progress_reset_count == 2
    assert not io.executed


def test_target_reference_rejects_drop_zone_in_wrist_without_stopping() -> None:
    io = FakeIO()
    controller, planner = _controller(io)

    with pytest.raises(ValueError, match="column view"):
        controller.set_target_reference(
            "drop_zone",
            "wrist",
            np.asarray([110.0, 70.0]),
            radius_px=24.0,
        )

    assert not planner.target_references
    assert not controller.stopped
    assert not io.executed


def test_replay_rejects_start_without_a_complete_next_chunk() -> None:
    sessions = {
        view: SimpleNamespace(frame_count=60)
        for view in ("wrist", "column")
    }
    with pytest.raises(ValueError, match="complete next five-frame chunk"):
        ReplaySO101IO(sessions, start_index=55)
