from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.evaluate_so101_ik_lewm_shadow import (
    _causal_freeze_history,
    _load_column_gripper_localization,
    _load_live_anchor_calibration,
    _normalized_session_action_chunk,
)
from scripts.visualize_so101_lewm import checkpoint_contract


def test_causal_freeze_history_uses_only_older_frames() -> None:
    frames = np.arange(3, dtype=np.uint8).reshape(3, 1, 1, 1)
    fallback = np.asarray([[[9]]], dtype=np.uint8)
    expected = {
        0: [0, 1, 2],
        1: [0, 1, 1],
        2: [0, 0, 0],
        3: [9, 9, 9],
    }
    for suffix, values in expected.items():
        frozen = _causal_freeze_history(frames, fallback, suffix)
        assert frozen.reshape(-1).tolist() == values
    assert frames.reshape(-1).tolist() == [0, 1, 2]


def test_normalized_session_action_chunk_prefers_recorded_commands() -> None:
    states = np.zeros((8, 6), dtype=np.float32)
    actions = np.arange(48, dtype=np.float32).reshape(8, 6)
    session = SimpleNamespace(states=states, actions=actions)
    result = _normalized_session_action_chunk(
        session,
        start=1,
        frameskip=5,
        center=np.zeros(6, dtype=np.float32),
        scale=np.full(6, 10.0, dtype=np.float32),
    )
    np.testing.assert_allclose(result.reshape(5, 6), actions[1:6] / 10.0)


def test_normalized_session_action_chunk_falls_back_to_state() -> None:
    states = np.arange(48, dtype=np.float32).reshape(8, 6)
    session = SimpleNamespace(states=states, actions=None)
    result = _normalized_session_action_chunk(
        session,
        start=2,
        frameskip=5,
        center=np.ones(6, dtype=np.float32),
        scale=np.full(6, 2.0, dtype=np.float32),
    )
    np.testing.assert_allclose(result.reshape(5, 6), (states[2:7] - 1.0) / 2.0)


def test_checkpoint_contract_falls_back_from_empty_args_cache(tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text("{}\n", encoding="utf-8")
    checkpoint_path = tmp_path / "memory.pt"
    torch.save(
        {
            "model": {},
            "args": {"cache": []},
            "run_config": {
                "contract": {
                    "caches": [str(cache)],
                    "frameskip": 5,
                    "history": 3,
                }
            },
        },
        checkpoint_path,
    )
    _, _, caches, frameskip, history, _ = checkpoint_contract(
        checkpoint_path, None
    )
    assert caches == [cache.resolve()]
    assert frameskip == 5
    assert history == 3


def test_load_column_gripper_localization_validates_live_center(tmp_path) -> None:
    path = tmp_path / "localization.json"
    path.write_text(
        '{"format":"so101-active-gripper-localization-v1",'
        '"camera":"column","center_px":[70.5,82.25],'
        '"opening_span":75.0,"max_arm_motion":0.2,'
        '"moving_area_px":300,"contrast":40.0,"elongation":3.0}\n',
        encoding="utf-8",
    )
    payload = _load_column_gripper_localization(path)
    np.testing.assert_allclose(payload["center_px"], [70.5, 82.25])


def test_load_live_anchor_calibration_validates_v9_contract(tmp_path) -> None:
    path = tmp_path / "live_anchor_calibration.json"
    path.write_text(
        """{
  "version": 1,
  "view": "column",
  "moving_tip_track_candidates_px": [
    [[70.0, 90.0], [96.0, 78.0]],
    [[63.0, 25.0], [70.0, 36.0]]
  ],
  "open_state": [1, 2, 3, 4, 5, 31],
  "closed_state": [1, 2, 3, 4, 5, 2]
}
""",
        encoding="utf-8",
    )

    payload = _load_live_anchor_calibration(path)

    assert payload["view"] == "column"
    assert payload["moving_tip_track_candidates_px"].shape == (2, 2, 2)
    np.testing.assert_array_equal(payload["open_state"], [1, 2, 3, 4, 5, 31])
    assert payload["source"] == str(path.resolve())


def test_load_live_anchor_calibration_rejects_wrong_camera(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        '{"version":1,"view":"wrist","moving_tip_track_candidates_px":[], '
        '"open_state":[],"closed_state":[]}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported live-anchor"):
        _load_live_anchor_calibration(path)
