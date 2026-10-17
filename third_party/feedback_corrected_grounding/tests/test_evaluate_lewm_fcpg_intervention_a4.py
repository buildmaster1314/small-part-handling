from __future__ import annotations

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from scripts import evaluate_lewm_fcpg_future_a4 as future_eval
from scripts.evaluate_lewm_fcpg_intervention_a4 import (
    _masked_input_epe,
    _paired_bootstrap,
)
from scripts.evaluate_lewm_fcpg_future_a4 import build_future_windows
from scripts.train_lewm_fcpg_paired_invariance_a4 import PairedFrames


def test_masked_input_epe_averages_only_common_valid_anchors() -> None:
    grounded = torch.tensor([[[0.0, 0.0], [3.0, 4.0], [9.0, 9.0]]])
    oracle = torch.zeros_like(grounded)
    valid = torch.tensor([[True, True, False]])
    assert _masked_input_epe(grounded, oracle, valid).item() == pytest.approx(2.5)


def test_episode_bootstrap_reports_nominal_minus_fcpg() -> None:
    nominal = np.asarray([3.0, 3.0, 5.0, 5.0])
    fcpg = np.asarray([1.0, 1.0, 2.0, 2.0])
    tasks = np.asarray(["a", "a", "b", "b"])
    episodes = np.asarray([0, 1, 0, 1])
    result = _paired_bootstrap(
        nominal,
        fcpg,
        tasks,
        episodes,
        draws=100,
        seed=7,
    )
    assert result["nominal_minus_fcpg"] == pytest.approx(2.5)
    assert result["relative_improvement"] == pytest.approx(0.625)
    assert result["ci95"] == pytest.approx([2.5, 2.5])
    assert result["positive_task_count"] == 2


def test_future_windows_are_causal_and_remain_inside_episode() -> None:
    frames = SimpleNamespace(
        task_name=np.asarray(["task"] * 50),
        episode_id=np.zeros(50, dtype=np.int64),
        frame_id=np.arange(50, dtype=np.int64),
        camera_step_frame=np.full(50, 20, dtype=np.int64),
    )
    current, future, action = build_future_windows(
        frames,
        horizons=2,
        post_step_only=True,
        settle_frames=1,
    )
    assert current[0] == 21
    assert future[0].tolist() == [26, 31]
    assert action[0].tolist() == [list(range(21, 26)), list(range(26, 31))]
    assert current[-1] == 39


def test_clean_view_uses_clean_rgb_and_clean_corrected_grounding(tmp_path) -> None:
    count = 2
    dynamic = np.zeros((count, 4, 4, 3), dtype=np.uint8)
    clean = np.full_like(dynamic, 7)
    nominal = np.zeros((count, 3, 2), dtype=np.float32)
    corrected = np.full_like(nominal, 2.5)
    task_name = np.asarray(["reach-v3"] * count)
    episode_id = np.zeros(count, dtype=np.int64)
    frame_id = np.arange(count, dtype=np.int64)
    frames = PairedFrames(
        dynamic_rgb=dynamic,
        clean_rgb=clean,
        proprio=np.zeros((count, 4), dtype=np.float32),
        q_nominal=nominal,
        q_fcpg=np.ones_like(nominal),
        q_true=np.full_like(nominal, 9.0),
        keypoints_xyz=np.zeros((count, 3, 3), dtype=np.float32),
        valid_nominal=np.ones((count, 3), dtype=bool),
        valid_true=np.zeros((count, 3), dtype=bool),
        raw_action=np.zeros((count, 4), dtype=np.float32),
        task_index=np.zeros(count, dtype=np.int64),
        task_name=task_name,
        episode_id=episode_id,
        frame_id=frame_id,
        camera_step_frame=np.full(count, -1, dtype=np.int64),
        image_size=(4, 4),
    )
    cache = tmp_path / "cache.npz"
    np.savez_compressed(
        cache,
        q_clean_corrected=corrected,
        task_name=task_name,
        episode_id=episode_id,
        frame_id=frame_id,
    )

    selected = future_eval._clean_view_frames(frames, cache)

    np.testing.assert_array_equal(selected.dynamic_rgb, clean)
    np.testing.assert_array_equal(selected.q_fcpg, corrected)
    np.testing.assert_array_equal(selected.q_true, nominal)
    np.testing.assert_array_equal(selected.valid_true, frames.valid_nominal)


def test_fcpg_deadzone_suppresses_only_small_frame_level_corrections() -> None:
    nominal = np.zeros((3, 3, 2), dtype=np.float32)
    corrected = nominal.copy()
    corrected[0, :, 0] = 3.0
    corrected[1, :, 0] = 5.0
    corrected[2, :, 0] = np.asarray([3.0, 4.0, 5.0], dtype=np.float32)
    count = len(nominal)
    frames = PairedFrames(
        dynamic_rgb=np.zeros((count, 4, 4, 3), dtype=np.uint8),
        clean_rgb=np.zeros((count, 4, 4, 3), dtype=np.uint8),
        proprio=np.zeros((count, 4), dtype=np.float32),
        q_nominal=nominal,
        q_fcpg=corrected,
        q_true=nominal.copy(),
        keypoints_xyz=np.zeros((count, 3, 3), dtype=np.float32),
        valid_nominal=np.ones((count, 3), dtype=bool),
        valid_true=np.ones((count, 3), dtype=bool),
        raw_action=np.zeros((count, 4), dtype=np.float32),
        task_index=np.zeros(count, dtype=np.int64),
        task_name=np.asarray(["a", "a", "b"]),
        episode_id=np.arange(count, dtype=np.int64),
        frame_id=np.zeros(count, dtype=np.int64),
        camera_step_frame=np.zeros(count, dtype=np.int64),
        image_size=(4, 4),
    )

    selected, active, magnitude = future_eval._apply_fcpg_deadzone(frames, 4.0)

    np.testing.assert_array_equal(active, np.asarray([False, True, False]))
    np.testing.assert_allclose(magnitude, np.asarray([3.0, 5.0, 4.0]))
    np.testing.assert_array_equal(selected.q_nominal, nominal)
    np.testing.assert_array_equal(selected.q_fcpg[0], nominal[0])
    np.testing.assert_array_equal(selected.q_fcpg[1], corrected[1])
    np.testing.assert_array_equal(selected.q_fcpg[2], nominal[2])

    summary = future_eval._deadzone_summary(
        frames,
        active,
        magnitude,
        threshold_px=4.0,
    )
    assert summary["activation_rate"] == pytest.approx(1.0 / 3.0)
    assert summary["per_task"]["a"]["activation_rate"] == pytest.approx(0.5)
    assert summary["per_task"]["b"]["activation_rate"] == pytest.approx(0.0)


@pytest.mark.parametrize("threshold", [-1.0, float("nan"), float("inf")])
def test_fcpg_deadzone_rejects_invalid_threshold(threshold: float) -> None:
    frames = SimpleNamespace(
        q_nominal=np.zeros((1, 3, 2), dtype=np.float32),
        q_fcpg=np.zeros((1, 3, 2), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="finite and non-negative"):
        future_eval._apply_fcpg_deadzone(frames, threshold)


def test_future_evaluation_keeps_full_horizon_for_metric_readout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.readout_shapes: list[tuple[int, ...]] = []

        def dynamics(
            self,
            state: torch.Tensor,
            action_prefix: torch.Tensor,
            *,
            task_index: torch.Tensor,
        ) -> torch.Tensor:
            del action_prefix, task_index
            return torch.zeros(len(state), 8, 4, 192)

        def metric_readout(
            self,
            predicted_state: torch.Tensor,
            current_state: torch.Tensor,
            current_q_px: torch.Tensor,
            task_index: torch.Tensor,
            *,
            image_size: tuple[int, int],
        ) -> torch.Tensor:
            del current_state, current_q_px, task_index, image_size
            self.readout_shapes.append(tuple(predicted_state.shape))
            assert predicted_state.shape[1:] == (8, 4, 192)
            return torch.zeros(len(predicted_state), 8, 3, 2)

    count = 16
    frames = SimpleNamespace(
        task_name=np.asarray(["task"] * count),
        episode_id=np.zeros(count, dtype=np.int64),
        frame_id=np.arange(count, dtype=np.int64),
        camera_step_frame=np.zeros(count, dtype=np.int64),
        task_index=np.zeros(count, dtype=np.int64),
        q_true=np.zeros((count, 3, 2), dtype=np.float32),
        valid_true=np.ones((count, 3), dtype=bool),
        raw_action=np.zeros((count, 4), dtype=np.float32),
        image_size=(224, 224),
    )

    def fake_frame_tensors(
        frames: SimpleNamespace,
        rows: np.ndarray,
        *,
        image_key: str,
        grounding: str,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        del frames, image_key, grounding
        return {
            "q_grounded": torch.zeros(len(rows), 3, 2, device=device),
            "valid": torch.ones(len(rows), 3, dtype=torch.bool, device=device),
        }

    monkeypatch.setattr(future_eval, "_frame_tensors", fake_frame_tensors)
    monkeypatch.setattr(
        future_eval,
        "_encode_frames",
        lambda _model, tensors: torch.zeros(
            len(tensors["q_grounded"]), 4, 192, device=tensors["q_grounded"].device
        ),
    )
    model = DummyModel()
    result = future_eval._evaluate(
        model,
        frames,
        horizons=2,
        post_step_only=False,
        settle_frames=0,
        device=torch.device("cpu"),
        batch_size=4,
        bootstrap_draws=10,
        seed=7,
    )
    assert result["horizons"] == 2
    assert model.readout_shapes
    assert all(shape[1:] == (8, 4, 192) for shape in model.readout_shapes)
