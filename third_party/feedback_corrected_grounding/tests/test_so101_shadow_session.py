import json

import numpy as np
import pytest

from fcpg.so101_kinematics import SO101_JOINT_NAMES
from fcpg.so101_shadow_session import (
    load_shadow_session,
    observation_rgb,
    observation_state,
    ordered_action,
    save_shadow_session,
)
from scripts.capture_so101_shadow_session import save_captured_sessions


def test_observation_and_action_use_fixed_joint_order():
    observation = {name: float(index) for index, name in enumerate(reversed(SO101_JOINT_NAMES))}
    observation["front"] = np.zeros((8, 12, 3), dtype=np.uint8)
    expected = np.asarray([observation[name] for name in SO101_JOINT_NAMES], dtype=np.float32)
    np.testing.assert_array_equal(observation_state(observation), expected)
    np.testing.assert_array_equal(ordered_action(observation), expected)
    assert observation_rgb(observation, "front").shape == (8, 12, 3)


def test_observation_rejects_missing_or_non_rgb_data():
    with pytest.raises(KeyError, match="missing joints"):
        observation_state({})
    with pytest.raises(ValueError, match=r"shape \[H, W, 3\]"):
        observation_rgb({"front": np.zeros((8, 8), dtype=np.uint8)}, "front")


def test_shadow_session_round_trip(tmp_path):
    frames = np.zeros((4, 16, 16, 3), dtype=np.uint8)
    states = np.arange(24, dtype=np.float32).reshape(4, 6)
    timestamps = np.arange(4, dtype=np.float64) / 30
    actions = states + 1
    saved = save_shadow_session(
        tmp_path / "session",
        frames=frames,
        states=states,
        timestamps_s=timestamps,
        actions=actions,
        source="test",
        camera="front",
        fps=30,
        action_source="test",
    )
    assert saved.frame_count == 4
    assert saved.manifest["control_sent_by_recorder"] is False
    np.testing.assert_array_equal(saved.states, states)
    np.testing.assert_array_equal(saved.actions, actions)

    manifest_path = saved.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["control_sent_by_recorder"] = True
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="no control"):
        load_shadow_session(saved.root)
    controlled = load_shadow_session(saved.root, allow_control=True)
    assert controlled.manifest["control_sent_by_recorder"] is True


def test_shadow_session_rejects_non_monotonic_timestamps(tmp_path):
    with pytest.raises(ValueError, match="strictly increasing"):
        save_shadow_session(
            tmp_path / "bad",
            frames=np.zeros((3, 8, 8, 3), dtype=np.uint8),
            states=np.zeros((3, 6), dtype=np.float32),
            timestamps_s=np.asarray([0.0, 0.1, 0.1]),
            actions=None,
            source="test",
            camera="front",
            fps=30,
            action_source="none",
        )


def test_paired_shadow_capture_saves_aligned_camera_sessions(tmp_path):
    output = tmp_path / "paired"
    states = [np.full(6, index, dtype=np.float32) for index in range(3)]
    frames = {
        "wrist": [np.full((8, 8, 3), index, dtype=np.uint8) for index in range(3)],
        "column": [np.full((8, 8, 3), index + 10, dtype=np.uint8) for index in range(3)],
    }
    save_captured_sessions(
        output,
        images_by_camera=frames,
        camera_fps={"wrist": 60, "column": 30},
        states=states,
        actions=[],
        timestamps=[0.0, 0.04, 0.08],
        sample_fps=25.0,
        robot_id="follower",
        leader_id=None,
    )

    wrist = load_shadow_session(output / "wrist")
    column = load_shadow_session(output / "column")
    np.testing.assert_array_equal(wrist.states, column.states)
    np.testing.assert_array_equal(wrist.timestamps_s, column.timestamps_s)
    assert wrist.manifest["camera_device_fps"] == 60
    assert column.manifest["camera_device_fps"] == 30
    assert wrist.manifest["control_sent_by_recorder"] is False
    assert column.manifest["control_sent_by_recorder"] is False
