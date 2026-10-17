from __future__ import annotations

from collections import deque

import numpy as np
import pytest
import torch

from fcpg.fk_deform_attn_baseline import PolicyReadoutOutput
import scripts.evaluate_fk_deform_attn_policy as evaluator


class _FrameRenderer:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = iter(frames)
        self.calls = 0

    def render(self) -> np.ndarray:
        self.calls += 1
        return next(self.frames)


def test_stable_initial_render_waits_for_exact_consecutive_frame() -> None:
    frames = [
        np.full((2, 3, 3), value, dtype=np.uint8)
        for value in (0, 1, 2, 2)
    ]
    env = _FrameRenderer(frames)

    result = evaluator.stable_initial_render(env, max_attempts=4)

    np.testing.assert_array_equal(result, frames[-1])
    assert env.calls == 4


def test_stable_initial_render_fails_closed_when_frames_keep_changing() -> None:
    env = _FrameRenderer(
        [
            np.full((2, 3, 3), value, dtype=np.uint8)
            for value in (0, 1, 2, 3)
        ]
    )

    with pytest.raises(RuntimeError, match="did not stabilize"):
        evaluator.stable_initial_render(env, max_attempts=4)


def _diagnostic_record(
    q_true: np.ndarray,
    valid: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "diagnostic_q_true": np.asarray(q_true, dtype=np.float32),
        "diagnostic_q_true_valid": np.asarray(valid, dtype=np.bool_),
    }


def _readout(
    sample_points_px: torch.Tensor,
    attention_weights: torch.Tensor,
) -> PolicyReadoutOutput:
    flat_count, keypoints, _, _ = sample_points_px.shape
    return PolicyReadoutOutput(
        weighted_visual=torch.zeros(flat_count, keypoints, 1),
        offsets_px=torch.zeros_like(sample_points_px),
        sample_points_px=sample_points_px,
        attention_weights=attention_weights,
        keypoint_latent=torch.zeros(flat_count, keypoints, 1),
        shared_residual_px=None,
    )


def _good_coordinate_inputs() -> tuple[PolicyReadoutOutput, torch.Tensor, torch.Tensor]:
    points = torch.tensor(
        [
            [[[0.0, 0.0], [4.0, 0.0]], [[0.0, 0.0], [0.0, 4.0]]],
            [[[1.0, 1.0], [5.0, 1.0]], [[0.0, 0.0], [0.0, 8.0]]],
        ]
    )
    weights = torch.tensor(
        [
            [[0.25, 0.75], [0.5, 0.5]],
            [[0.5, 0.5], [0.25, 0.75]],
        ]
    )
    q_true = torch.tensor(
        [[
            [[0.0, 0.0], [float("nan"), float("nan")]],
            [[0.0, 1.0], [0.0, 2.0]],
        ]]
    )
    valid = torch.tensor([[[True, False], [True, True]]])
    return _readout(points, weights), q_true, valid


def test_diagnostic_history_batch_pads_oldest_record_and_preserves_targets() -> None:
    first_q = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    second_q = np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    first_valid = np.asarray([True, False])
    second_valid = np.asarray([False, True])
    history = deque(
        [
            _diagnostic_record(first_q, first_valid),
            _diagnostic_record(second_q, second_valid),
        ]
    )

    q_true, valid = evaluator.diagnostic_history_batch(
        history,
        obs_horizon=3,
        device=torch.device("cpu"),
    )

    assert q_true.shape == (1, 3, 2, 2)
    assert q_true.dtype == torch.float32
    assert valid.shape == (1, 3, 2)
    assert valid.dtype == torch.bool
    torch.testing.assert_close(
        q_true[0],
        torch.from_numpy(np.stack((first_q, first_q, second_q))),
    )
    torch.testing.assert_close(
        valid[0],
        torch.from_numpy(np.stack((first_valid, first_valid, second_valid))),
    )
    assert len(history) == 2


def test_diagnostic_history_batch_rejects_empty_or_incomplete_history() -> None:
    with pytest.raises(ValueError, match="history must not be empty"):
        evaluator.diagnostic_history_batch(
            deque(),
            obs_horizon=2,
            device=torch.device("cpu"),
        )

    incomplete = deque(
        [
            _diagnostic_record(
                np.zeros((1, 2), dtype=np.float32),
                np.ones(1, dtype=bool),
            ),
            {"diagnostic_q_true": np.zeros((1, 2), dtype=np.float32)},
        ]
    )
    with pytest.raises(KeyError, match="actual-camera projection targets"):
        evaluator.diagnostic_history_batch(
            incomplete,
            obs_horizon=2,
            device=torch.device("cpu"),
        )


class _RecordingReadoutModel:
    def __init__(self, readout: PolicyReadoutOutput) -> None:
        self.readout = readout
        self.seen_keys: tuple[str, ...] | None = None
        self.eval_calls = 0

    def eval(self) -> "_RecordingReadoutModel":
        self.eval_calls += 1
        return self

    def forward_with_readout(self, batch: dict[str, torch.Tensor]):
        self.seen_keys = tuple(batch)
        return "action-output", self.readout


def test_coordinate_diagnostic_keeps_q_true_out_of_model_input() -> None:
    readout = _readout(
        torch.tensor([[[[1.0, 2.0], [1.0, 2.0]]]]),
        torch.tensor([[[0.5, 0.5]]]),
    )
    model = _RecordingReadoutModel(readout)
    batch = {
        "rgb": torch.zeros(1, 1, 3, 4, 4),
        "obs": torch.zeros(1, 1, 2),
        "keypoints_xyz": torch.zeros(1, 1, 1, 3),
        "q_nominal": torch.zeros(1, 1, 1, 2),
    }
    q_true = torch.tensor([[[[1.0, 2.0]]]])
    valid = torch.ones(1, 1, 1, dtype=torch.bool)

    output, diagnostic = evaluator.predict_action_chunk_with_coordinate_diagnostic(
        model,
        batch,
        q_true,
        valid,
    )

    assert output == "action-output"
    assert model.seen_keys == tuple(batch)
    assert "q_true" not in model.seen_keys
    assert "diagnostic_q_true" not in model.seen_keys
    assert model.eval_calls == 1
    assert diagnostic["epe_claim_supported"] is True


@pytest.mark.parametrize("forbidden_key", ("q_true", "diagnostic_q_true"))
def test_projection_targets_are_rejected_if_added_to_model_batch(
    forbidden_key: str,
) -> None:
    with pytest.raises(RuntimeError, match="must not enter"):
        evaluator.predict_action_chunk_with_coordinate_diagnostic(
            object(),
            {forbidden_key: torch.zeros(1)},
            torch.zeros(1, 1, 1, 2),
            torch.ones(1, 1, 1, dtype=torch.bool),
        )


def test_coordinate_diagnostic_uses_attention_barycenter_epe_and_valid_mask() -> None:
    readout, q_true, valid = _good_coordinate_inputs()

    result = evaluator._coordinate_diagnostic_from_readout(readout, q_true, valid)

    assert result["status"] == "available"
    assert result["name"] == "attention_barycenter_epe_px"
    assert result["epe_claim_supported"] is True
    assert result["reason"] == ""
    assert result["latest_epe_sum_px"] == pytest.approx(7.0)
    assert result["latest_valid_count"] == 2
    assert result["all_epe_sum_px"] == pytest.approx(10.0)
    assert result["all_valid_count"] == 3


def _assert_unavailable(result: dict[str, object], reason: str) -> None:
    assert result["status"] == "unavailable"
    assert result["name"] == "attention_barycenter_epe_px"
    assert result["epe_claim_supported"] is False
    assert reason in str(result["reason"])
    assert result["latest_epe_sum_px"] is None
    assert result["latest_valid_count"] == 0
    assert result["all_epe_sum_px"] is None
    assert result["all_valid_count"] == 0


@pytest.mark.parametrize(
    ("readout", "q_true", "valid", "reason"),
    (
        (
            object(),
            torch.zeros(1, 1, 1, 2),
            torch.ones(1, 1, 1, dtype=torch.bool),
            "PolicyReadoutOutput",
        ),
        (
            _readout(
                torch.tensor([[[[0.0, 0.0], [float("nan"), 0.0]]]]),
                torch.tensor([[[0.5, 0.5]]]),
            ),
            torch.zeros(1, 1, 1, 2),
            torch.ones(1, 1, 1, dtype=torch.bool),
            "non-finite",
        ),
        (
            _readout(
                torch.zeros(1, 1, 2, 2),
                torch.tensor([[[0.25, 0.25]]]),
            ),
            torch.zeros(1, 1, 1, 2),
            torch.ones(1, 1, 1, dtype=torch.bool),
            "not normalized",
        ),
        (
            _readout(
                torch.zeros(2, 1, 2, 2),
                torch.full((2, 1, 2), 0.5),
            ),
            torch.zeros(1, 2, 1, 2),
            torch.tensor([[[True], [False]]]),
            "no valid actual-camera",
        ),
    ),
)
def test_coordinate_diagnostic_fails_closed(
    readout: object,
    q_true: torch.Tensor,
    valid: torch.Tensor,
    reason: str,
) -> None:
    _assert_unavailable(
        evaluator._coordinate_diagnostic_from_readout(readout, q_true, valid),
        reason,
    )


def test_physical_hash_is_view_independent_and_view_hash_is_pair_specific() -> None:
    observation = np.arange(5, dtype=np.float32)
    keypoints = np.arange(6, dtype=np.float32).reshape(2, 3)
    physical_inputs = (
        ("task", "pick-place-v3"),
        ("task_index", 12),
        ("env_seed", 34),
        ("initial_observation", observation),
        ("initial_keypoints_xyz", keypoints),
    )

    physical_hash = evaluator._hash_arrays(*physical_inputs)
    same_physical_hash = evaluator._hash_arrays(
        ("task", "pick-place-v3"),
        ("task_index", 12),
        ("env_seed", 34),
        ("initial_observation", observation.copy()),
        ("initial_keypoints_xyz", keypoints.copy()),
    )
    changed_physical_hash = evaluator._hash_arrays(
        *physical_inputs[:-1],
        ("initial_keypoints_xyz", keypoints + 1.0),
    )

    assert len(physical_hash) == 64
    assert physical_hash == same_physical_hash
    assert physical_hash != changed_physical_hash

    view_inputs = (
        ("physical_episode_sha256", physical_hash),
        ("camera_translation", np.asarray([0.01, 0.0, 0.0])),
        ("camera_axis", np.asarray([0.0, 1.0, 0.0])),
        ("camera_rotation", 2.0),
        ("initial_rgb", np.zeros((3, 2, 2), dtype=np.float32)),
        ("initial_q_nominal", np.zeros((2, 2), dtype=np.float32)),
        ("initial_q_true", np.ones((2, 2), dtype=np.float32)),
        ("initial_q_true_valid", np.ones(2, dtype=np.bool_)),
    )
    paired_view_hash = evaluator._hash_arrays(*view_inputs)
    same_paired_view_hash = evaluator._hash_arrays(
        ("physical_episode_sha256", physical_hash),
        ("camera_translation", np.asarray([0.01, 0.0, 0.0]).copy()),
        ("camera_axis", np.asarray([0.0, 1.0, 0.0]).copy()),
        ("camera_rotation", 2.0),
        ("initial_rgb", np.zeros((3, 2, 2), dtype=np.float32)),
        ("initial_q_nominal", np.zeros((2, 2), dtype=np.float32)),
        ("initial_q_true", np.ones((2, 2), dtype=np.float32)),
        ("initial_q_true_valid", np.ones(2, dtype=np.bool_)),
    )
    changed_camera_view_hash = evaluator._hash_arrays(
        *view_inputs[:1],
        ("camera_translation", np.asarray([0.02, 0.0, 0.0])),
        *view_inputs[2:],
    )
    changed_physical_view_hash = evaluator._hash_arrays(
        ("physical_episode_sha256", changed_physical_hash),
        *view_inputs[1:],
    )

    assert len(paired_view_hash) == 64
    assert paired_view_hash == same_paired_view_hash
    assert paired_view_hash != changed_camera_view_hash
    assert paired_view_hash != changed_physical_view_hash
