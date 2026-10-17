import threading
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from fcpg.so101_kinematics import SO101_JOINT_NAMES
from fcpg.so101_realtime_runtime import LiveObservation
from scripts.serve_so101_realtime_fcpg_lewm import (
    HardwareSO101IO,
    RoundArtifactWriter,
    _compose_live_mujoco_frame,
    _jpeg_bytes,
    _load_home_state_report,
    _preview_safe_state,
    _request_error_code,
    _target_overlay_label,
    page,
)
import scripts.serve_so101_realtime_fcpg_lewm as realtime_server
from fcpg.so101_realtime_runtime import PlanRejectedError


def test_target_overlay_uses_semantic_roles() -> None:
    assert _target_overlay_label("pickup") == "PICKUP TARGET"
    assert _target_overlay_label("drop_zone") == "DROP ZONE"
    assert _target_overlay_label("legacy_object_class") == "TARGET"


def _live(state: np.ndarray, value: int) -> LiveObservation:
    return LiveObservation(
        timestamp_s=float(value),
        state=np.asarray(state, dtype=np.float32),
        frames={
            "wrist": np.full((224, 224, 3), value, dtype=np.uint8),
            "column": np.full((224, 224, 3), value, dtype=np.uint8),
        },
    )


class _Follower:
    def __init__(self) -> None:
        self.bus = _Bus()
        self.cameras = {
            "wrist": _Camera(35),
            "column": _Camera(175),
        }

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        return {name: action[name] for name in SO101_JOINT_NAMES}


class _Bus:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.writes: list[tuple[str, dict[str, float], int]] = []

    def sync_read(self, register: str, *, num_retry: int) -> dict[str, float]:
        self.calls.append((register, num_retry))
        return {
            name.removesuffix(".pos"): float(index)
            for index, name in enumerate(SO101_JOINT_NAMES)
        }

    def sync_write(
        self,
        register: str,
        values: dict[str, float],
        *,
        num_retry: int,
    ) -> None:
        self.writes.append((register, dict(values), num_retry))


class _Camera:
    def __init__(self, value: int) -> None:
        self.value = value
        self.read_count = 0

    def async_read(self, timeout_ms: float = 1000) -> np.ndarray:
        assert timeout_ms > 0
        self.read_count += 1
        return np.full((480, 640, 3), self.value, dtype=np.uint8)


class _RecoveringCamera:
    def __init__(self, value: int, *, always_fail: bool = False) -> None:
        self.value = value
        self.always_fail = always_fail
        self.connected = True
        self.read_count = 0
        self.connect_count = 0
        self.disconnect_count = 0
        self.latest_frame = np.full((480, 640, 3), 7, dtype=np.uint8)
        self.new_frame_event = threading.Event()
        self.new_frame_event.set()
        self.frame_lock = threading.Lock()

    def async_read(self, timeout_ms: float = 1000) -> np.ndarray:
        assert timeout_ms > 0
        self.read_count += 1
        if self.always_fail or self.connect_count == 0:
            raise TimeoutError("stale camera handle")
        return np.full((480, 640, 3), self.value, dtype=np.uint8)

    def disconnect(self) -> None:
        self.disconnect_count += 1
        self.connected = False

    def connect(self, *, warmup: bool = True) -> None:
        assert warmup is False
        self.connect_count += 1
        self.connected = True


class _PulseFollower:
    def __init__(self, initial: float, *, follows_commands: bool = True) -> None:
        self.state = np.asarray([1, 2, 3, 4, 5, initial], dtype=np.float32)
        self.follows_commands = follows_commands
        self.commands: list[dict[str, float]] = []
        self.bus = _PulseBus(self)


class _PulseBus:
    def __init__(self, follower: _PulseFollower) -> None:
        self.follower = follower

    def sync_write(
        self,
        register: str,
        values: dict[str, float],
        *,
        num_retry: int,
    ) -> None:
        assert register == "Goal_Position"
        assert set(values) == {"gripper"}
        assert num_retry == realtime_server.STATE_WRITE_RETRIES
        command = {"gripper.pos": float(values["gripper"])}
        self.follower.commands.append(command)
        if self.follower.follows_commands:
            self.follower.state[5] = command["gripper.pos"]


def _pulse_io(follower: _PulseFollower) -> HardwareSO101IO:
    io = HardwareSO101IO.__new__(HardwareSO101IO)
    io.follower = follower
    io.gripper_calibration_closed = 2.0
    io.gripper_calibration_open = 32.0

    def observe() -> LiveObservation:
        value = int(round(float(follower.state[5])))
        return _live(follower.state.copy(), value)

    io.observe = observe
    return io


def test_preview_clipping_uses_a_copy() -> None:
    state = np.asarray([101.0, -103.0, 2.0, 3.0, 4.0, -2.0], dtype=np.float32)
    original = state.copy()

    safe, clipped = _preview_safe_state(state)

    assert clipped
    np.testing.assert_array_equal(state, original)
    np.testing.assert_array_equal(safe, [100.0, -100.0, 2.0, 3.0, 4.0, 0.0])


def test_live_preview_matches_frozen_real_and_predicted_layout() -> None:
    real = np.full((224, 224, 3), 40, dtype=np.uint8)
    predicted = np.full((480, 570, 3), 160, dtype=np.uint8)
    command_path = np.asarray(
        [[0.0, 0.0, 0.1], [0.01, 0.02, 0.1]], dtype=np.float64
    )
    prediction_path = np.asarray(
        [[0.0, 0.0, 0.1], [0.02, 0.01, 0.1]], dtype=np.float64
    )

    frame = _compose_live_mujoco_frame(
        real,
        predicted,
        phase="approach",
        candidate_index=3,
        horizon=1,
        raw_ahead=5,
        command_xyz=command_path[1],
        predicted_xyz=prediction_path[1],
        command_path=command_path,
        prediction_path=prediction_path,
        predicted_clipped=False,
    )

    assert frame.size == (1280, 800)
    html = page()
    assert "Frozen real wrist observation vs LeWM MuJoCo prediction" in html
    assert "MuJoCo robot-only safety preview" not in html
    assert "复制错误" in html
    assert "showRequestError" in html
    assert "copyDiagnostics" in html
    assert "复制完整诊断" in html
    assert "标定夹爪位置" in html
    assert "执行 5 步" in html
    assert "/api/calibrate" in html
    assert "requestInFlight" in html
    assert "if(confirm('执行当前规划的 5 个小步？'))" not in html
    assert (
        "post('/api/execute',{round_id:latest.plan.round_id,confirmed:true}" in html
    )
    assert "last_execution_batch" in html
    assert "setAsset('mujoco'" in html
    assert "/stream/wrist.mjpg" in html
    assert "/stream/column.mjpg" in html
    assert "not model history" in html
    assert "monitor_frames_feed_model" in html
    assert "/api/target-reference" in html
    assert "/api/object-prior" not in html
    assert "抓取目标" in html
    assert "放置区域" in html
    assert "click fish" not in html
    assert "panel planning" in html
    assert "setAsset('liveWrist'" not in html
    assert "setAsset('liveColumn'" not in html
    assert "setInterval(refresh,1500)" not in html


def test_home_state_report_ignores_stale_image_coordinates(tmp_path: Path) -> None:
    report = tmp_path / "legacy-report.json"
    report.write_text(
        '{"current_state":[1,2,3,4,5,6],'
        '"object_phase_guard":{"current":{"centroid_px":[9,10]}},'
        '"current_goal":{"descriptor":{"centroid_px":[11,12]}}}',
        encoding="utf-8",
    )

    state = _load_home_state_report(report)

    np.testing.assert_array_equal(state, [1, 2, 3, 4, 5, 6])


def test_monitor_frame_encodes_as_jpeg() -> None:
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    frame[..., 0] = 180

    payload = _jpeg_bytes(frame)

    assert payload.startswith(b"\xff\xd8")
    with Image.open(BytesIO(payload)) as image:
        assert image.size == (224, 224)
        assert image.mode == "RGB"


def test_request_errors_have_stable_frontend_codes() -> None:
    assert _request_error_code(
        RuntimeError("There is no status packet")
    ) == "robot_communication"
    assert _request_error_code(
        RuntimeError("Timed out waiting for frame from camera")
    ) == "camera_timeout"
    assert _request_error_code(
        PlanRejectedError("live state drift requires a fresh plan")
    ) == "plan_rejected"
    assert _request_error_code(ValueError("bad payload")) == "invalid_request"
    assert _request_error_code(TypeError("bad radius")) == "invalid_request"


def test_column_overlay_keeps_current_and_target_above_paths() -> None:
    current = np.asarray([80.0, 90.0], dtype=np.float32)
    target = np.asarray([25.0, 150.0], dtype=np.float32)
    plan = SimpleNamespace(
        frames={"column": np.zeros((3, 224, 224, 3), dtype=np.uint8)},
        corrected={
            "column": np.asarray(
                [[[0.0, 0.0]] * 3, [[0.0, 0.0]] * 3, [current, [84, 90], [76, 90]]],
                dtype=np.float32,
            )
        },
        target={"visible": True, "kind": "red_object", "centroid_px": target},
        commanded_gripper_px=np.repeat(current[None], 8, axis=0),
        predicted_gripper_px=np.repeat(current[None], 8, axis=0),
    )

    overlay = np.asarray(RoundArtifactWriter._target_overlay(plan))

    np.testing.assert_array_equal(overlay[90, 80], [255, 255, 255])
    assert overlay[150, 25, 0] > 200
    assert overlay[150, 25, 1] > 150


def test_column_overlay_draws_all_three_fk_anchor_predictions() -> None:
    current = np.asarray([[20, 20], [20, 50], [20, 80]], dtype=np.float32)
    horizons = np.linspace(30.0, 100.0, 8, dtype=np.float32)
    predicted = np.stack(
        [
            np.stack((horizons, np.full(8, y, dtype=np.float32)), axis=-1)
            for y in (20.0, 50.0, 80.0)
        ],
        axis=1,
    )
    plan = SimpleNamespace(
        frames={"column": np.zeros((3, 224, 224, 3), dtype=np.uint8)},
        corrected={"column": np.repeat(current[None], 3, axis=0)},
        target={"visible": False},
        commanded_gripper_px=None,
        predicted_gripper_px=predicted[:, 0],
        commanded_anchor_px=None,
        predicted_anchor_px=predicted,
    )

    overlay = np.asarray(RoundArtifactWriter._target_overlay(plan))

    for y, color in (
        (20, [22, 163, 74]),
        (50, [46, 211, 255]),
        (80, [255, 92, 185]),
    ):
        row = overlay[y, 30:101]
        assert np.any(np.all(row == color, axis=-1))


def test_mujoco_preview_failure_does_not_discard_plan_artifacts(
    tmp_path, monkeypatch
) -> None:
    anchors = np.asarray(
        [[[100.0, 100.0], [95.0, 100.0], [105.0, 100.0]]] * 3,
        dtype=np.float32,
    )
    plan = SimpleNamespace(
        guard=SimpleNamespace(round_id="round-preview-failure"),
        frames={
            view: np.zeros((3, 224, 224, 3), dtype=np.uint8)
            for view in ("wrist", "column")
        },
        nominal={view: anchors.copy() for view in ("wrist", "column")},
        corrected={view: anchors.copy() for view in ("wrist", "column")},
        valid={
            view: np.ones((3, 3), dtype=bool) for view in ("wrist", "column")
        },
        target={"visible": False},
        commanded_gripper_px=None,
        predicted_gripper_px=None,
        selected_index=0,
    )
    writer = RoundArtifactWriter(tmp_path, render_mujoco=True)
    monkeypatch.setattr(
        writer,
        "_mujoco_gif",
        lambda *_: (_ for _ in ()).throw(RuntimeError("EGL_BAD_ACCESS")),
    )
    monkeypatch.setattr(
        "scripts.serve_so101_realtime_fcpg_lewm.plan_json", lambda _: {}
    )

    assets = writer(plan)

    assert "mujoco_preview" not in assets
    assert "mujoco_preview_error" in assets
    error_path = tmp_path / "rounds/round-preview-failure/mujoco_preview_error.txt"
    assert "EGL_BAD_ACCESS" in error_path.read_text(encoding="utf-8")
    assert (tmp_path / "rounds/round-preview-failure/plan.json").is_file()


def test_hardware_state_read_retries_without_polling_cameras() -> None:
    io = HardwareSO101IO.__new__(HardwareSO101IO)
    io.follower = _Follower()
    io.max_command_jump = 10.0

    state = io._read_state_only()

    np.testing.assert_array_equal(state, np.arange(6, dtype=np.float32))
    assert io.follower.bus.calls == [
        ("Present_Position", realtime_server.STATE_READ_RETRIES)
    ]


def test_hardware_monitor_uses_cached_frame_while_planning_owns_cameras() -> None:
    io = HardwareSO101IO.__new__(HardwareSO101IO)
    io.follower = _Follower()
    io.camera_lock = threading.Lock()
    io.preview_cache_lock = threading.Lock()
    io.preview_cache = {}

    observation = io.observe()
    assert observation.frames["wrist"].shape == (224, 224, 3)
    assert observation.frames["column"].shape == (224, 224, 3)
    assert io.follower.cameras["wrist"].read_count == 1
    assert io.follower.cameras["column"].read_count == 1

    io.camera_lock.acquire()
    try:
        cached = io.preview_frame("wrist")
    finally:
        io.camera_lock.release()

    np.testing.assert_array_equal(cached, observation.frames["wrist"])
    assert io.follower.cameras["wrist"].read_count == 1


def test_hardware_observation_recovers_only_failed_camera() -> None:
    io = HardwareSO101IO.__new__(HardwareSO101IO)
    io.follower = _Follower()
    io.follower.cameras["wrist"] = _RecoveringCamera(91)
    io.camera_lock = threading.Lock()
    io.preview_cache_lock = threading.Lock()
    io.preview_cache = {}
    io.camera_reconnect_counts = {view: 0 for view in realtime_server.VIEWS}
    io.camera_last_errors = {view: None for view in realtime_server.VIEWS}

    observation = io.observe()

    assert np.all(observation.frames["wrist"] == 91)
    assert np.all(observation.frames["column"] == 175)
    wrist = io.follower.cameras["wrist"]
    assert wrist.connect_count == 1
    assert wrist.disconnect_count == 1
    assert wrist.read_count == 2
    assert wrist.latest_frame is None
    assert not wrist.new_frame_event.is_set()
    assert io.follower.cameras["column"].read_count == 1
    assert io.camera_reconnect_counts == {"wrist": 1, "column": 0}


def test_hardware_observation_fails_closed_after_bounded_camera_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(realtime_server, "CAMERA_RECOVERY_DELAY_S", 0.0)
    io = HardwareSO101IO.__new__(HardwareSO101IO)
    io.follower = _Follower()
    failed = _RecoveringCamera(91, always_fail=True)
    io.follower.cameras["wrist"] = failed
    io.camera_lock = threading.Lock()
    io.preview_cache_lock = threading.Lock()
    io.preview_cache = {
        "wrist": np.full((224, 224, 3), 33, dtype=np.uint8)
    }
    io.camera_reconnect_counts = {view: 0 for view in realtime_server.VIEWS}
    io.camera_last_errors = {view: None for view in realtime_server.VIEWS}

    with pytest.raises(TimeoutError, match="did not recover after 3"):
        io.observe()

    assert failed.connect_count == realtime_server.CAMERA_RECOVERY_ATTEMPTS
    assert failed.read_count == 1 + realtime_server.CAMERA_RECOVERY_ATTEMPTS
    assert not failed.connected
    assert io.camera_reconnect_counts["wrist"] == 0
    assert io.camera_last_errors["wrist"] == "stale camera handle"
    np.testing.assert_array_equal(io.preview_cache["wrist"], 33)
    assert io.follower.cameras["column"].read_count == 0


def test_hardware_chunk_waits_for_tracking_and_stability() -> None:
    io = HardwareSO101IO.__new__(HardwareSO101IO)
    io.follower = _Follower()
    io.max_command_jump = 10.0
    io.settle_timeout_s = 0.5
    io.settle_position_tolerance = 2.0
    io.settle_step_tolerance = 0.25
    io.settle_samples = 2
    target = np.full(6, 5.0, dtype=np.float32)
    states = [
        np.zeros(6, dtype=np.float32),
        np.full(6, 2.5, dtype=np.float32),
        np.full(6, 2.75, dtype=np.float32),
        np.full(6, 3.0, dtype=np.float32),
        np.full(6, 4.5, dtype=np.float32),
        np.full(6, 4.75, dtype=np.float32),
        np.full(6, 4.9, dtype=np.float32),
    ]
    iterator = iter(states)
    state_reads = 0
    camera_reads = 0

    def read_state_only() -> np.ndarray:
        nonlocal state_reads
        state_reads += 1
        return next(iterator)

    def observe() -> LiveObservation:
        nonlocal camera_reads
        camera_reads += 1
        return _live(np.full(6, 4.9, dtype=np.float32), camera_reads)

    io._read_state_only = read_state_only
    io.observe = observe
    chunk = np.stack([target] * 5)

    result = io.execute_chunk(
        chunk,
        period_s=1e-6,
        stop_event=threading.Event(),
    )

    assert len(result.observations) == 5
    np.testing.assert_allclose(result.observations[-1].state, 4.9)
    assert state_reads == len(states)
    assert camera_reads == 1
    assert len(io.follower.bus.writes) == 5
    assert all(
        register == "Goal_Position"
        and retry == realtime_server.STATE_WRITE_RETRIES
        and set(values) == {name.removesuffix(".pos") for name in SO101_JOINT_NAMES}
        for register, values, retry in io.follower.bus.writes
    )


def test_gripper_calibration_uses_observed_data_range_only(
    monkeypatch,
) -> None:
    follower = _PulseFollower(32.0)
    io = _pulse_io(follower)
    monkeypatch.setattr(realtime_server.time, "sleep", lambda _: None)

    def localize(_frames, states, **_kwargs):
        closed = int(np.argmin(states[:, 5]))
        opened = int(np.argmax(states[:, 5]))
        return SimpleNamespace(
            center_px=(72.0, 88.0),
            open_tip_px=(82.0, 92.0),
            moving_tip_track_candidates_px=(
                ((76.0, 28.0), (88.0, 28.0)),
                ((70.0, 94.0), (98.0, 94.0)),
            ),
            closed_frame_index=closed,
            open_frame_index=opened,
            to_dict=lambda: {
                "center_px": [72.0, 88.0],
                "open_tip_px": [82.0, 92.0],
                "moving_tip_track_candidates_px": [
                    [[76.0, 28.0], [88.0, 28.0]],
                    [[70.0, 94.0], [98.0, 94.0]],
                ],
                "closed_frame_index": closed,
                "open_frame_index": opened,
                "opening_span": float(np.ptp(states[:, 5])),
            },
        )

    monkeypatch.setattr(realtime_server, "localize_gripper_from_pulse", localize)

    result = io.calibrate_gripper(stop_event=threading.Event())

    values = np.asarray([row["gripper.pos"] for row in follower.commands])
    assert len(values) > 0
    assert values[0] == 30.5
    assert np.max(np.abs(np.diff(values))) <= 1.5
    assert np.min(values) >= 2.0
    assert np.max(values) <= 32.0
    assert result["closed_position"] == 2.0
    assert result["open_position"] == 32.0
    assert result["initial_position"] == 32.0
    assert result["final_position"] == pytest.approx(32.0, abs=2.0)
    assert result["restored_initial"] is True
    assert result["observed_gripper_range"] == [2.0, 32.0]
    assert result["other_joint_commands_sent"] is False
    assert [row["name"] for row in result["phase_ranges"]] == [
        "close",
        "open",
        "restore",
    ]
    assert follower.commands[-1]["gripper.pos"] == pytest.approx(32.0, abs=2.0)
    assert follower.state[5] == pytest.approx(32.0, abs=2.0)
    assert set(result["views"]) == {"column"}
    np.testing.assert_allclose(result["views"]["column"]["open_tip_px"], [82.0, 92.0])
    np.testing.assert_allclose(
        result["views"]["column"]["moving_tip_track_candidates_px"],
        [
            [[76.0, 28.0], [88.0, 28.0]],
            [[70.0, 94.0], [98.0, 94.0]],
        ],
    )


def test_gripper_calibration_capture_persists_replayable_frames(tmp_path: Path) -> None:
    frames = {
        "wrist": np.stack(
            [np.full((24, 32, 3), value, dtype=np.uint8) for value in (10, 20, 30)]
        ),
        "column": np.stack(
            [np.full((24, 32, 3), value, dtype=np.uint8) for value in (40, 50, 60)]
        ),
    }
    states = np.zeros((3, 6), dtype=np.float32)
    states[:, 5] = [30.0, 2.0, 30.0]
    commands = np.asarray([28.5, 3.5, 30.0], dtype=np.float32)
    phases = [
        {"name": "open", "start": 0, "stop": 1},
        {"name": "close", "start": 1, "stop": 2},
        {"name": "reopen", "start": 2, "stop": 3},
    ]

    result = realtime_server._save_gripper_calibration_capture(
        tmp_path,
        frames,
        states,
        commands,
        phases,
        sample_hz=20.0,
        closed_position=2.0,
        open_position=30.0,
    )

    with np.load(result["npz"]) as payload:
        np.testing.assert_array_equal(payload["wrist_frames"], frames["wrist"])
        np.testing.assert_array_equal(payload["column_frames"], frames["column"])
        np.testing.assert_array_equal(payload["states"], states)
        np.testing.assert_array_equal(payload["commands"], commands)
    with Image.open(result["dual_gif"]) as animation:
        assert animation.n_frames == 3
    html = Path(result["index_html"]).read_text(encoding="utf-8")
    assert 'src="dual_pulse.gif"' in html
    assert Path(result["metadata"]).is_file()


def test_live_anchor_calibration_round_trips_through_fk_validated_track(
    tmp_path: Path,
) -> None:
    tracks = np.asarray(
        [[[70.0, 90.0], [96.0, 78.0]], [[63.0, 25.0], [70.0, 36.0]]],
        dtype=np.float32,
    )
    open_state = np.asarray([1, 2, 3, 4, 5, 31], dtype=np.float32)
    closed_state = np.asarray([1, 2, 3, 4, 5, 2], dtype=np.float32)
    calibration = {
        "views": {
            "column": {
                "moving_tip_track_candidates_px": tracks,
                "open_state": open_state,
                "closed_state": closed_state,
            }
        },
        "geometry": {"column": {"candidate_index": 0}},
        "capture": {"npz": "/tmp/pulse.npz"},
    }
    path = realtime_server._save_live_anchor_calibration(tmp_path, calibration)

    class Planner:
        def __init__(self) -> None:
            self.model_contract = {}
            self.arguments = None

        def set_live_moving_tip_track(
            self,
            view,
            candidates,
            opened,
            closed,
        ):
            self.arguments = (view, candidates, opened, closed)
            return {"candidate_index": 0}

    planner = Planner()
    result = realtime_server._load_live_anchor_calibration(planner, path)

    assert result == {"candidate_index": 0}
    view, loaded_tracks, loaded_open, loaded_closed = planner.arguments
    assert view == "column"
    np.testing.assert_array_equal(loaded_tracks, tracks)
    np.testing.assert_array_equal(loaded_open, open_state)
    np.testing.assert_array_equal(loaded_closed, closed_state)
    assert planner.model_contract["live_anchor_calibration_source"] == str(path)


def test_gripper_calibration_timeout_reports_observed_motion(monkeypatch) -> None:
    follower = _PulseFollower(2.0, follows_commands=False)
    io = _pulse_io(follower)
    monkeypatch.setattr(realtime_server.time, "sleep", lambda _: None)

    with pytest.raises(
        RuntimeError,
        match=(
            r"phase open: target=32\.000, start=2\.000, final=2\.000, "
            r"observed_range=\[2\.000,2\.000\], last_command=3\.500"
        ),
    ):
        io.calibrate_gripper(stop_event=threading.Event())

    assert len(follower.commands) == 106
    assert all(row["gripper.pos"] == 2.0 for row in follower.commands[:6])
    assert all(row["gripper.pos"] == 3.5 for row in follower.commands[6:])
    assert all(set(row) == {"gripper.pos"} for row in follower.commands)
