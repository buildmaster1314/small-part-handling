from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.render_lewm_bridge_prediction_gallery import (
    _latent_result,
    _render_animation,
    _render_panel,
)


def _sample() -> dict[str, object]:
    rgb = torch.zeros((4, 16, 16, 3), dtype=torch.uint8)
    rgb[:, :, :, 0] = torch.arange(4, dtype=torch.uint8).view(4, 1, 1) * 40
    points = torch.tensor(
        [[[3.0, 3.0], [8.0, 8.0], [12.0, 12.0]]] * 4,
        dtype=torch.float32,
    )
    return {
        "rgb": rgb,
        "frame_id": torch.tensor([10, 11, 12, 13]),
        "q_nominal": points,
        "valid_nominal": torch.ones((4, 3), dtype=torch.bool),
        "source": "push",
        "episode_id": "held-out-1",
        "shuffled_episode_id": "held-out-2",
    }


def _animation_result(offset: float) -> dict[str, object]:
    target = np.zeros((3, 4), dtype=np.float32)
    vectors = {
        "target": target,
        "factual": target + offset,
        "copy": target + 0.5,
        "shuffled": target - offset,
    }
    return {"sequence": [_latent_result(vectors) for _ in range(3)]}


def test_latent_result_computes_per_anchor_and_aggregate_metrics() -> None:
    result = _latent_result(
        {
            "target": np.zeros((3, 2), dtype=np.float64),
            "factual": np.array([[1, -1], [2, 0], [0, 0]]),
            "copy": np.array([[2, 0], [0, 2], [1, 1]]),
            "shuffled": np.array([[-1, 1], [0, 0], [2, 0]]),
        }
    )

    np.testing.assert_allclose(result["per_anchor_mse"]["factual"], [1, 2, 0])
    np.testing.assert_allclose(result["per_anchor_mse"]["copy"], [2, 2, 1])
    np.testing.assert_allclose(result["per_anchor_mse"]["shuffled"], [1, 0, 2])
    assert result["mse"] == pytest.approx(
        {"factual": 1.0, "copy": 5.0 / 3.0, "shuffled": 1.0}
    )
    np.testing.assert_allclose(result["per_anchor_action_response_mse"], [4, 2, 2])
    assert result["action_response_mse"] == pytest.approx(8.0 / 3.0)
    assert result["action_response_over_copy_mse"] == pytest.approx(8.0 / 5.0)
    assert all(vector.dtype == np.float32 for vector in result["vectors"].values())


def test_render_animation_produces_three_sized_palette_frames() -> None:
    frames = _render_animation(
        _sample(),
        {
            "tiny": _animation_result(0.25),
            "bridge": _animation_result(0.125),
        },
    )

    assert len(frames) == 3
    for frame in frames:
        assert frame.size == (1_080, 885)
        assert frame.mode == "P"
        assert frame.palette is not None
        assert frame.getcolors(maxcolors=160) is not None


@pytest.mark.parametrize("transition_index", [-1, 3])
def test_render_panel_rejects_out_of_bounds_transition_index(
    transition_index: int,
) -> None:
    with pytest.raises(ValueError, match="outside the observed sequence"):
        _render_panel(_sample(), {}, transition_index=transition_index)
