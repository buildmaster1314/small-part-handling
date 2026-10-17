from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from fcpg.mujoco_camera import set_fixed_camera_pose_offset
from fcpg.mujoco_camera_trajectory import (
    anchor_observation_schedule,
    trajectory_frame,
    trajectory_phase,
    trajectory_supervision_masks,
)
import scripts.collect_metaworld_dynamic_camera_dataset as dynamic_collector
from scripts.collect_metaworld_dynamic_camera_dataset import _validated_args


def _endpoint() -> tuple[np.ndarray, np.ndarray, float]:
    return (
        np.array([0.04, -0.02, 0.01], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        -5.0,
    )


def test_step_is_clean_then_persists_at_one_sampled_endpoint() -> None:
    endpoint = _endpoint()
    frames = [
        trajectory_frame("step", index, 5, endpoint, step_fraction=0.5)
        for index in range(5)
    ]

    assert [frame.phase for frame in frames] == [0.0, 0.0, 1.0, 1.0, 1.0]
    for frame in frames[:2]:
        np.testing.assert_array_equal(frame.translation_world, np.zeros(3))
        assert frame.rotation_degrees == 0.0
    for frame in frames[2:]:
        np.testing.assert_array_equal(frame.translation_world, endpoint[0])
        np.testing.assert_array_equal(frame.rotation_axis_world, endpoint[1])
        assert frame.rotation_degrees == endpoint[2]


def test_linear_drift_is_continuous_and_reaches_same_endpoint() -> None:
    endpoint = _endpoint()
    frames = [
        trajectory_frame(
            "linear_drift",
            index,
            5,
            endpoint,
            drift_start_fraction=0.25,
            drift_end_fraction=0.75,
        )
        for index in range(5)
    ]

    np.testing.assert_allclose(
        [frame.phase for frame in frames],
        [0.0, 0.0, 0.5, 1.0, 1.0],
        rtol=0.0,
        atol=0.0,
    )
    for frame in frames:
        np.testing.assert_allclose(
            frame.translation_world,
            endpoint[0] * frame.phase,
            rtol=0.0,
            atol=1e-12,
        )
        assert frame.rotation_degrees == endpoint[2] * frame.phase
        np.testing.assert_array_equal(frame.rotation_axis_world, endpoint[1])
    np.testing.assert_array_equal(endpoint[0], np.array([0.04, -0.02, 0.01]))


def test_supervision_masks_only_unpredictable_camera_transitions() -> None:
    episode = trajectory_supervision_masks("episode", np.ones(5))
    np.testing.assert_array_equal(episode.intervention_onset, np.zeros(5, dtype=bool))
    np.testing.assert_array_equal(
        episode.prior_supervision_valid, np.ones(5, dtype=bool)
    )

    frame = trajectory_supervision_masks("frame", np.ones(5))
    np.testing.assert_array_equal(frame.intervention_onset, np.ones(5, dtype=bool))
    np.testing.assert_array_equal(
        frame.prior_supervision_valid, np.zeros(5, dtype=bool)
    )

    step = trajectory_supervision_masks("step", np.array([0, 0, 1, 1, 1]))
    np.testing.assert_array_equal(
        step.intervention_onset, np.array([False, False, True, False, False])
    )
    np.testing.assert_array_equal(
        step.prior_supervision_valid, np.array([True, True, False, True, True])
    )

    drift = trajectory_supervision_masks(
        "linear_drift", np.array([0.0, 0.0, 0.5, 1.0, 1.0])
    )
    np.testing.assert_array_equal(
        drift.intervention_onset, np.array([False, False, True, False, False])
    )
    np.testing.assert_array_equal(
        drift.prior_supervision_valid,
        np.array([True, True, False, True, False]),
    )


def test_anchor_dropout_marks_local_withholding_and_recovery() -> None:
    onset = np.array(
        [False, False, True, False, False, False, False, False, False, False]
    )
    schedule = anchor_observation_schedule(
        10,
        3,
        onset,
        anchor_indices=(1,),
        start_offset=0,
        length=4,
        minimum_recovery_frames=4,
    )

    expected_active = np.zeros((10, 3), dtype=bool)
    expected_active[2:6, 1] = True
    expected_recovery_onset = np.zeros((10, 3), dtype=bool)
    expected_recovery_onset[6, 1] = True
    expected_recovery = np.zeros((10, 3), dtype=bool)
    expected_recovery[6:, 1] = True
    np.testing.assert_array_equal(schedule.dropout_active, expected_active)
    np.testing.assert_array_equal(schedule.recovery_onset, expected_recovery_onset)
    np.testing.assert_array_equal(schedule.recovery_segment, expected_recovery)


@pytest.mark.parametrize("length", [1, 3, 7])
def test_anchor_dropout_rejects_lengths_outside_four_to_six(length: int) -> None:
    with pytest.raises(ValueError, match=r"\[4, 6\]"):
        anchor_observation_schedule(
            12,
            3,
            np.array([False, True] + [False] * 10),
            anchor_indices=(0,),
            length=length,
        )


def test_anchor_dropout_allows_explicit_recursive_length() -> None:
    schedule = anchor_observation_schedule(
        40,
        3,
        np.array([False, True] + [False] * 38),
        anchor_indices=(0,),
        start_offset=0,
        length=25,
        maximum_length=25,
        minimum_recovery_frames=4,
    )
    assert int(schedule.dropout_active[:, 0].sum()) == 25
    assert bool(schedule.recovery_onset[26, 0])


def test_anchor_dropout_supports_a_finite_recovery_window() -> None:
    schedule = anchor_observation_schedule(
        20,
        3,
        np.array([False, False, True] + [False] * 17),
        anchor_indices=(1,),
        start_offset=0,
        length=4,
        minimum_recovery_frames=4,
        recovery_length=3,
    )
    expected = np.zeros((20, 3), dtype=bool)
    expected[6:9, 1] = True
    np.testing.assert_array_equal(schedule.recovery_segment, expected)


def test_anchor_dropout_rejects_nonpositive_finite_recovery() -> None:
    with pytest.raises(ValueError, match="recovery_length"):
        anchor_observation_schedule(
            12,
            3,
            np.array([False, True] + [False] * 10),
            anchor_indices=(0,),
            length=4,
            recovery_length=0,
        )


def test_anchor_dropout_rejects_invalid_explicit_maximum() -> None:
    with pytest.raises(ValueError, match="at least 4"):
        anchor_observation_schedule(
            12,
            3,
            np.array([False, True] + [False] * 10),
            anchor_indices=(0,),
            length=4,
            maximum_length=3,
        )


def test_anchor_dropout_requires_a_recovery_segment() -> None:
    with pytest.raises(ValueError, match="recovery"):
        anchor_observation_schedule(
            8,
            3,
            np.array([False, False, True, False, False, False, False, False]),
            anchor_indices=(0,),
            start_offset=1,
            length=4,
            minimum_recovery_frames=2,
        )


def test_anchor_dropout_rejects_zero_recovery_frames() -> None:
    with pytest.raises(ValueError, match="positive"):
        anchor_observation_schedule(
            6,
            3,
            np.array([False, False, True, False, False, False]),
            anchor_indices=(0,),
            start_offset=0,
            length=4,
            minimum_recovery_frames=0,
        )


def test_anchor_dropout_allows_one_boundary_recovery_frame() -> None:
    schedule = anchor_observation_schedule(
        7,
        3,
        np.array([False, False, True, False, False, False, False]),
        anchor_indices=(0,),
        start_offset=0,
        length=4,
        minimum_recovery_frames=1,
    )

    expected_onset = np.zeros((7, 3), dtype=bool)
    expected_onset[6, 0] = True
    expected_segment = np.zeros((7, 3), dtype=bool)
    expected_segment[6:, 0] = True
    np.testing.assert_array_equal(schedule.recovery_onset, expected_onset)
    np.testing.assert_array_equal(schedule.recovery_segment, expected_segment)


@pytest.mark.parametrize("mode", ["episode", "frame"])
def test_legacy_modes_realize_complete_supplied_offset(mode: str) -> None:
    endpoint = _endpoint()
    frame = trajectory_frame(mode, 1, 3, endpoint)
    assert frame.phase == 1.0
    np.testing.assert_array_equal(frame.translation_world, endpoint[0])
    np.testing.assert_array_equal(frame.rotation_axis_world, endpoint[1])
    assert frame.rotation_degrees == endpoint[2]


def test_real_mujoco_helper_realizes_world_translation_and_rotation_phase() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><camera name="corner" pos="1 2 3" '
        'quat="0.9238795325 0.3826834324 0 0"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base_position = model.cam_pos[0].copy()
    base_quaternion = model.cam_quat[0].copy()
    base_rotation = data.cam_xmat[0].reshape(3, 3).copy()
    endpoint = (
        np.array([0.04, -0.02, 0.01]),
        np.array([0.0, 0.0, 1.0]),
        10.0,
    )
    frame = trajectory_frame("linear_drift", 2, 5, endpoint)
    assert frame.phase == 0.5

    set_fixed_camera_pose_offset(
        model,
        data,
        0,
        base_position,
        base_quaternion,
        *frame.as_pose_offset(),
    )
    np.testing.assert_allclose(
        data.cam_xpos[0] - base_position, endpoint[0] * 0.5, atol=1e-12
    )
    rotation = data.cam_xmat[0].reshape(3, 3)
    relative = rotation @ base_rotation.T
    relative_angle = math.degrees(
        math.acos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    )
    assert relative_angle == pytest.approx(5.0, abs=1e-8)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-12)
    assert np.linalg.norm(model.cam_quat[0]) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mode": "unknown"}, "unknown camera trajectory"),
        ({"frame_index": -1}, "frame_index"),
        ({"frame_index": 5}, "frame_index"),
        ({"frame_count": 0}, "frame_count"),
        ({"step_fraction": float("nan")}, "step_fraction"),
        (
            {"drift_start_fraction": 0.8, "drift_end_fraction": 0.2},
            "drift_start_fraction",
        ),
    ],
)
def test_trajectory_phase_rejects_invalid_contract(
    kwargs: dict[str, object], match: str
) -> None:
    arguments: dict[str, object] = {
        "mode": "step",
        "frame_index": 0,
        "frame_count": 5,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=match):
        trajectory_phase(**arguments)


def test_dynamic_collector_cli_accepts_all_modes_and_freezes_nominal_design() -> None:
    for mode in ("episode", "frame", "step", "linear_drift"):
        args = _validated_args(
            [
                "--actual-camera-trajectory",
                mode,
                "--steps-per-episode",
                "5",
                "--independent-rng-streams",
            ]
        )
        assert args.actual_camera_trajectory == mode
        assert args.steps_per_episode == 5
        assert args.independent_rng_streams is True

    legacy = _validated_args(
        ["--actual-camera-randomization", "frame"]
    )
    assert legacy.actual_camera_trajectory == "frame"


def test_dynamic_collector_cli_accepts_random_step_and_local_withholding() -> None:
    args = _validated_args(
        [
            "--actual-camera-trajectory",
            "step",
            "--actual-camera-step-fraction-min",
            "0.25",
            "--actual-camera-step-fraction-max",
            "0.4",
            "--anchor-dropout-mode",
            "measurement",
            "--anchor-dropout-indices",
            "1",
            "--anchor-dropout-length",
            "4",
            "--steps-per-episode",
            "16",
            "--paired-clean-view",
            "--independent-rng-streams",
        ]
    )
    assert args.actual_camera_step_fraction_min == 0.25
    assert args.actual_camera_step_fraction_max == 0.4
    assert args.anchor_dropout_index_list == [1]
    assert args.paired_clean_view is True
    earliest_phases = np.asarray(
        [
            trajectory_phase(
                "step",
                frame_index,
                args.steps_per_episode,
                step_fraction=args.actual_camera_step_fraction_min,
            )
            for frame_index in range(args.steps_per_episode)
        ]
    )
    earliest_onset = np.flatnonzero(
        trajectory_supervision_masks("step", earliest_phases).intervention_onset
    )
    assert earliest_onset.tolist() == [args.minimum_history_size]


def test_dynamic_collector_cli_requires_explicit_long_dropout_maximum() -> None:
    common = [
        "--actual-camera-trajectory",
        "step",
        "--actual-camera-step-fraction",
        "0.2",
        "--anchor-dropout-mode",
        "visual",
        "--anchor-dropout-indices",
        "0",
        "--anchor-dropout-length",
        "25",
        "--anchor-dropout-start-offset",
        "10",
        "--steps-per-episode",
        "100",
        "--minimum-recovery-frames",
        "20",
        "--anchor-recovery-window-length",
        "10",
    ]
    with pytest.raises(SystemExit):
        _validated_args(common)

    args = _validated_args(
        common + ["--anchor-dropout-maximum-length", "25"]
    )
    assert args.anchor_dropout_length == 25
    assert args.anchor_dropout_maximum_length == 25
    assert args.anchor_recovery_window_length == 10


def test_legacy_short_collection_remains_compatible_without_observer_protocol() -> None:
    legacy = _validated_args(["--steps-per-episode", "1"])
    assert legacy.steps_per_episode == 1

    with pytest.raises(SystemExit):
        _validated_args(
            [
                "--steps-per-episode",
                "1",
                "--paired-clean-view",
            ]
        )


def test_random_step_timing_requires_independent_rng_streams() -> None:
    with pytest.raises(SystemExit):
        _validated_args(
            [
                "--actual-camera-trajectory",
                "step",
                "--actual-camera-step-fraction-min",
                "0.2",
                "--actual-camera-step-fraction-max",
                "0.4",
            ]
        )


def test_dynamic_collector_cli_rejects_zero_recovery_frames() -> None:
    with pytest.raises(SystemExit):
        _validated_args(["--minimum-recovery-frames", "0"])


def test_observer_protocol_rejects_step_bounds_without_clean_history() -> None:
    with pytest.raises(SystemExit):
        _validated_args(
            [
                "--actual-camera-trajectory",
                "step",
                "--actual-camera-step-fraction-min",
                "0.2",
                "--actual-camera-step-fraction-max",
                "0.4",
                "--steps-per-episode",
                "16",
                "--paired-clean-view",
                "--independent-rng-streams",
            ]
        )


def test_linear_drift_observer_requires_and_accepts_clean_history() -> None:
    with pytest.raises(SystemExit):
        _validated_args(
            [
                "--actual-camera-trajectory",
                "linear_drift",
                "--steps-per-episode",
                "16",
                "--paired-clean-view",
            ]
        )

    args = _validated_args(
        [
            "--actual-camera-trajectory",
            "linear_drift",
            "--actual-camera-drift-start-fraction",
            "0.3",
            "--steps-per-episode",
            "16",
            "--paired-clean-view",
        ]
    )
    assert args.actual_camera_drift_start_fraction == 0.3


def test_dynamic_collector_cli_rejects_invalid_drift_interval() -> None:
    with pytest.raises(SystemExit):
        _validated_args(
            [
                "--actual-camera-drift-start-fraction",
                "0.7",
                "--actual-camera-drift-end-fraction",
                "0.7",
            ]
        )


@pytest.mark.parametrize(
    "flag",
    [
        "--actual-camera-translation-cm-min",
        "--actual-camera-translation-cm-max",
        "--actual-camera-rotation-deg-min",
        "--actual-camera-rotation-deg-max",
    ],
)
def test_dynamic_collector_cli_rejects_nonfinite_camera_limits(flag: str) -> None:
    with pytest.raises(SystemExit):
        _validated_args([flag, "inf"])


def test_dynamic_collector_cli_rejects_duplicate_tasks_and_bad_output_suffix() -> None:
    with pytest.raises(SystemExit):
        _validated_args(["--tasks", "reach-v3,reach-v3"])
    with pytest.raises(SystemExit):
        _validated_args(["--output", "dynamic.dat"])


def test_dynamic_collector_a4_factory_resolves_native_task_cameras() -> None:
    args = _validated_args(
        [
            "--tasks",
            "pick-place-v3,drawer-retrieve-v0,towel-wipe-v0",
            "--camera",
            "a4-default",
            "--a4-task-factory",
            "--vary-task-index",
            "--task-index-offset",
            "36",
            "--episodes-per-task",
            "3",
        ]
    )

    assert dynamic_collector._camera_name(args, "pick-place-v3") == "corner"
    assert dynamic_collector._camera_name(args, "drawer-retrieve-v0") == "drawer_view"
    assert dynamic_collector._camera_name(args, "towel-wipe-v0") == "towel_view"


def test_dynamic_collector_a4_orbit_view_uses_each_tasks_base_camera() -> None:
    args = _validated_args(
        [
            "--tasks",
            "pick-place-v3,drawer-retrieve-v0,towel-wipe-v0",
            "--camera",
            "a4-default",
            "--a4-task-factory",
            "--a4-orbit-clock-hour",
            "3",
            "--a4-orbit-elevation-deg",
            "40",
            "--a4-orbit-radius-m",
            "1.25",
        ]
    )

    for task, expected_base in (
        ("pick-place-v3", "corner"),
        ("drawer-retrieve-v0", "drawer_view"),
        ("towel-wipe-v0", "towel_view"),
    ):
        view = dynamic_collector._orbit_camera_view(
            args, dynamic_collector._camera_name(args, task)
        )
        assert view is not None
        assert view.clock_hour == 3
        assert view.elevation_degrees == 40.0
        assert view.radius_m == 1.25
        assert view.base_camera_name == expected_base


@pytest.mark.parametrize(
    "argv",
    [
        ["--a4-orbit-clock-hour", "3"],
        [
            "--a4-task-factory",
            "--camera",
            "corner",
            "--a4-orbit-clock-hour",
            "3",
        ],
        [
            "--a4-task-factory",
            "--camera",
            "a4-default",
            "--a4-orbit-clock-hour",
            "0",
        ],
        [
            "--a4-task-factory",
            "--camera",
            "a4-default",
            "--a4-orbit-clock-hour",
            "3",
            "--a4-orbit-elevation-deg",
            "90",
        ],
        [
            "--a4-task-factory",
            "--camera",
            "a4-default",
            "--a4-orbit-clock-hour",
            "3",
            "--a4-orbit-radius-m",
            "0.1",
        ],
    ],
)
def test_dynamic_collector_a4_orbit_rejects_invalid_contract(argv) -> None:
    with pytest.raises(SystemExit):
        _validated_args(argv)


def test_dynamic_collector_a4_factory_rejects_unknown_tasks_and_instances() -> None:
    with pytest.raises(SystemExit):
        _validated_args(
            ["--tasks", "button-press-v3", "--a4-task-factory"]
        )
    with pytest.raises(SystemExit):
        _validated_args(
            [
                "--tasks",
                "reach-v3",
                "--a4-task-factory",
                "--vary-task-index",
                "--task-index-offset",
                "47",
                "--episodes-per-task",
                "2",
            ]
        )
    with pytest.raises(SystemExit):
        _validated_args(["--camera", "a4-default"])


def test_output_validator_rejects_empty_collection() -> None:
    args = _validated_args([])
    arrays = {name: np.empty((0,)) for name in dynamic_collector._FRAME_ARRAY_NAMES}
    arrays["rgb"] = np.empty((0, args.height, args.width, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="no recorded frames"):
        dynamic_collector._validate_collected_arrays(arrays, args)


def test_dynamic_collector_writes_step_endpoint_phase_and_fixed_nominal(
    tmp_path, monkeypatch
) -> None:
    class FakeEnv:
        def __init__(self) -> None:
            self.model = SimpleNamespace(
                cam_fovy=np.array([90.0]),
                cam_pos=np.zeros((1, 3), dtype=np.float64),
                cam_quat=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
            )
            self.data = SimpleNamespace(
                offset=(
                    np.zeros(3, dtype=np.float64),
                    np.array([0.0, 0.0, 1.0], dtype=np.float64),
                    0.0,
                )
            )
            self.action_space = object()
            self.value = 0.0
            self.step_camera_offsets: list[float] = []

        def reset(self, seed: int):
            self.value = 0.0
            self.step_camera_offsets = []
            return np.full(4, self.value, dtype=np.float32), {}

        def render(self):
            value = 100 if float(self.data.offset[0][0]) == 0.0 else 200
            return np.full((8, 8, 3), value, dtype=np.uint8)

        def step(self, action):
            self.step_camera_offsets.append(float(self.data.offset[0][0]))
            self.value += 1.0
            obs = np.full(4, self.value, dtype=np.float32)
            return obs, self.value, False, False, {"success": 0.0}

        def close(self) -> None:
            pass

    def fake_extrinsic(data, camera_id: int) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 3] = float(data.offset[0][0])
        return matrix

    def fake_set_pose(
        model,
        data,
        camera_id,
        base_position,
        base_quaternion,
        translation,
        axis,
        degrees,
    ) -> None:
        data.offset = (
            np.asarray(translation, dtype=np.float64).copy(),
            np.asarray(axis, dtype=np.float64).copy(),
            float(degrees),
        )

    endpoint = (
        np.array([0.04, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        5.0,
    )
    keypoints = SimpleNamespace(
        names=["left", "right", "tcp"],
        xyz=np.array(
            [[-0.1, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]],
            dtype=np.float64,
        ),
    )
    fake_env = FakeEnv()
    monkeypatch.setattr(dynamic_collector, "make_env", lambda *args, **kwargs: fake_env)
    monkeypatch.setattr(
        dynamic_collector,
        "make_scripted_policy",
        lambda task: object(),
    )
    monkeypatch.setattr(
        dynamic_collector,
        "choose_action",
        lambda action_space, rng, obs, policy: np.array(
            [obs[0], 0.0, 0.0, 0.0], dtype=np.float32
        ),
    )
    monkeypatch.setattr(dynamic_collector, "get_camera_id", lambda model, name: 0)
    monkeypatch.setattr(
        dynamic_collector, "camera_extrinsic_cv_from_mujoco", fake_extrinsic
    )
    monkeypatch.setattr(
        dynamic_collector, "set_fixed_camera_pose_offset", fake_set_pose
    )
    monkeypatch.setattr(
        dynamic_collector,
        "sample_camera_pose_offset",
        lambda *args, **kwargs: endpoint,
    )
    monkeypatch.setattr(
        dynamic_collector, "get_site_keypoints", lambda *args: keypoints
    )

    output = tmp_path / "dynamic.npz"
    args = _validated_args(
        [
            "--tasks",
            "reach-v3",
            "--width",
            "8",
            "--height",
            "8",
            "--episodes-per-task",
            "1",
            "--steps-per-episode",
            "5",
            "--warmup-steps",
            "1",
            "--actual-camera-trajectory",
            "step",
            "--actual-camera-step-fraction",
            "1.0",
            "--independent-rng-streams",
            "--paired-clean-view",
            "--output",
            str(output),
        ]
    )
    dynamic_collector.collect(args)

    with np.load(output, allow_pickle=False) as data:
        np.testing.assert_array_equal(
            data["actual_camera_trajectory_phase"],
            np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            data["actual_camera_translation_xyz"][:, 0],
            np.array([0.0, 0.0, 0.0, 0.0, 0.04]),
        )
        np.testing.assert_allclose(
            data["actual_camera_endpoint_translation_xyz"][:, 0], 0.04
        )
        np.testing.assert_array_equal(
            data["nominal_extrinsic"],
            np.repeat(np.eye(4, dtype=np.float32)[None], 5, axis=0),
        )
        np.testing.assert_array_equal(data["obs"][:, 0], np.arange(1.0, 6.0))
        np.testing.assert_array_equal(data["action"][:, 0], data["obs"][:, 0])
        assert data["observation_action_alignment"].item() == "pre_action"
        assert data["actual_camera_trajectory_mode"].item() == "step"
        assert data["nominal_extrinsic_schedule"].item() == "fixed_base_per_episode"
        np.testing.assert_array_equal(
            data["camera_intervention_onset"],
            np.array([False, False, False, False, True]),
        )
        np.testing.assert_array_equal(
            data["prior_supervision_valid"],
            np.array([True, True, True, True, False]),
        )
        np.testing.assert_array_equal(
            data["episode_recorded_frame_count"], np.full(5, 5, dtype=np.int32)
        )
        np.testing.assert_array_equal(
            data["measurement_valid"], data["valid_nominal"]
        )
        assert not bool(data["anchor_dropout_active"].any())
        assert data["paired_clean_rgb"].shape == (5, 8, 8, 3)
        assert data["paired_view_contract"].item().startswith(
            "same_pre_action_physical_state"
        )
        assert data["rgb_observation_contract"].item() == "actual_camera_unmodified"
        assert (
            data["measurement_valid_contract"].item()
            == "valid_nominal_and_not_anchor_dropout"
        )
        assert data["dataset_type"].item() == dynamic_collector.DYNAMIC_CAMERA_DATASET_TYPE
        assert int(data["schema_version"].item()) == 2
        assert data["camera_name"].item() == "corner"
        assert int(data["warmup_steps"].item()) == 1
        assert data["q_true_usage"].item() == "offline_target_and_metrics_only"
        config = json.loads(data["collection_config_json"].item())
        assert config["tasks"] == ["reach-v3"]
        assert config["actual_camera_trajectory"] == "step"
        assert config["warmup_steps"] == 1
        source_path = dynamic_collector.Path(data["collector_source"].item())
        assert source_path == dynamic_collector.Path(dynamic_collector.__file__).resolve()
        assert len(data["collector_source_sha256"].item()) == 64
    np.testing.assert_allclose(
        fake_env.step_camera_offsets, [0.0, 0.0, 0.0, 0.0, 0.0, 0.04]
    )

    dropout_output = tmp_path / "dynamic_dropout.npz"
    dropout_args = _validated_args(
        [
            "--tasks",
            "reach-v3",
            "--width",
            "8",
            "--height",
            "8",
            "--episodes-per-task",
            "1",
            "--steps-per-episode",
            "14",
            "--warmup-steps",
            "0",
            "--actual-camera-trajectory",
            "step",
            "--actual-camera-step-fraction",
            "0.3",
            "--anchor-dropout-mode",
            "measurement",
            "--anchor-dropout-indices",
            "1",
            "--anchor-dropout-start-offset",
            "1",
            "--anchor-dropout-length",
            "4",
            "--minimum-recovery-frames",
            "5",
            "--independent-rng-streams",
            "--output",
            str(dropout_output),
        ]
    )
    dynamic_collector.collect(dropout_args)
    with np.load(dropout_output, allow_pickle=False) as data:
        expected_dropout = np.zeros((14, 3), dtype=bool)
        expected_dropout[5:9, 1] = True
        expected_recovery_onset = np.zeros((14, 3), dtype=bool)
        expected_recovery_onset[9, 1] = True
        expected_recovery = np.zeros((14, 3), dtype=bool)
        expected_recovery[9:, 1] = True
        np.testing.assert_array_equal(data["anchor_dropout_active"], expected_dropout)
        np.testing.assert_array_equal(
            data["anchor_recovery_onset"], expected_recovery_onset
        )
        np.testing.assert_array_equal(
            data["anchor_recovery_segment"], expected_recovery
        )
        np.testing.assert_array_equal(
            data["measurement_valid"], data["valid_nominal"] & ~expected_dropout
        )

    visual_output = tmp_path / "dynamic_visual_dropout.npz"
    visual_args = _validated_args(
        [
            "--tasks",
            "reach-v3",
            "--width",
            "8",
            "--height",
            "8",
            "--episodes-per-task",
            "1",
            "--steps-per-episode",
            "14",
            "--warmup-steps",
            "0",
            "--actual-camera-trajectory",
            "step",
            "--actual-camera-step-fraction",
            "0.3",
            "--anchor-dropout-mode",
            "visual",
            "--anchor-dropout-indices",
            "1",
            "--anchor-dropout-start-offset",
            "1",
            "--anchor-dropout-length",
            "4",
            "--anchor-occlusion-radius-px",
            "1",
            "--minimum-recovery-frames",
            "5",
            "--paired-clean-view",
            "--independent-rng-streams",
            "--output",
            str(visual_output),
        ]
    )
    dynamic_collector.collect(visual_args)
    with np.load(visual_output, allow_pickle=False) as data:
        active = data["anchor_dropout_active"].any(axis=1)
        assert bool(active.any())
        np.testing.assert_array_equal(
            data["rgb"][~active], data["rgb_camera_only"][~active]
        )
        np.testing.assert_array_equal(
            data["paired_clean_rgb"][~active],
            data["paired_clean_rgb_unmodified"][~active],
        )
        assert bool(
            np.any(data["rgb"][active] != data["rgb_camera_only"][active])
        )
        assert bool(
            np.any(
                data["paired_clean_rgb"][active]
                != data["paired_clean_rgb_unmodified"][active]
            )
        )
        assert (
            data["paired_view_contract"].item()
            == "same_pre_action_physical_state_matched_anchor_visibility_"
            "dynamic_then_clean_rerender_v2"
        )
        assert (
            data["paired_clean_rgb_contract"].item()
            == "clean_camera_with_matched_visual_anchor_occlusion"
        )
        assert (
            data["paired_clean_rgb_unmodified_contract"].item()
            == "clean_camera_unmodified_raw_view"
        )
