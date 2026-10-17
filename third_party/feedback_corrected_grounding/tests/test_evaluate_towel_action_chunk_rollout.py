from __future__ import annotations

from collections import deque

import numpy as np
import torch

from fcpg.lewm_counterfactual_runtime import FrameRecord
from scripts.evaluate_towel_action_chunk_rollout import (
    append_phase_run,
    configure_grounding,
    history_batch,
    policy_frame_record,
)


def _record(value: float) -> FrameRecord:
    return FrameRecord(
        rgb=np.full((3, 8, 8), value, dtype=np.float32),
        proprio=np.full(9, value, dtype=np.float32),
        keypoints_xyz=np.full((3, 3), value, dtype=np.float32),
        q_nominal=np.full((3, 2), value, dtype=np.float32),
        valid_nominal=np.ones(3, dtype=bool),
        q_true=np.full((3, 2), value, dtype=np.float32),
        valid_true=np.ones(3, dtype=bool),
    )


def test_history_batch_left_pads_first_record() -> None:
    batch = history_batch(deque([_record(2.0)]), 2, torch.device("cpu"))
    assert batch["rgb"].shape == (1, 2, 3, 8, 8)
    assert torch.equal(batch["obs"][0, 0], batch["obs"][0, 1])
    assert float(batch["q_true"].mean()) == 2.0


def test_phase_runs_keep_only_wipe_transitions() -> None:
    runs: list[str] = []
    for phase in (
        "grasp_towel",
        "wipe_right",
        "wipe_right",
        "wipe_left",
        "lift_after_wipe",
    ):
        append_phase_run(runs, phase)
    assert runs == ["wipe_right", "wipe_left"]


def test_policy_frame_record_adds_only_robot_and_goal() -> None:
    record = _record(2.0)
    observation = np.arange(39, dtype=np.float32)

    selected = policy_frame_record(record, observation, "robot_goal")

    np.testing.assert_array_equal(
        selected.proprio,
        np.concatenate((observation[:4], observation[-3:])),
    )
    assert selected.rgb is record.rgb


def test_policy_frame_record_can_hold_a_deployment_goal_fixed() -> None:
    observation = np.arange(39, dtype=np.float32)
    selected = policy_frame_record(
        _record(2.0),
        observation,
        "robot_goal",
        np.asarray((0.1, 0.2, 0.3), dtype=np.float32),
    )

    np.testing.assert_allclose(selected.proprio[-3:], (0.1, 0.2, 0.3))


def test_configure_grounding_keeps_checkpoint_mode_unchanged() -> None:
    model = type("Model", (), {"mode": "geoprop"})()
    configure_grounding(model, "checkpoint", None, torch.device("cpu"))
    assert model.mode == "geoprop"


def test_configure_grounding_requires_checkpoint_for_fcpg() -> None:
    model = type("Model", (), {"mode": "geoprop"})()
    with np.testing.assert_raises_regex(ValueError, "requires an FCPG checkpoint"):
        configure_grounding(model, "fcpg", None, torch.device("cpu"))


def test_configure_grounding_can_select_oracle_without_adapter() -> None:
    class Model:
        mode = "geoprop"

        def set_reliability_override(self, value: str) -> None:
            self.reliability = value

        def eval(self) -> "Model":
            return self

    model = Model()
    configure_grounding(model, "oracle", None, torch.device("cpu"))
    assert model.mode == "oracle"
    assert model.reliability == "ones"
