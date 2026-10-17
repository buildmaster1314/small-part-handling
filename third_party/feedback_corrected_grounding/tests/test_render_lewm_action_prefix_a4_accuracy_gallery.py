from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from fcpg.lewm_action_branch_a4 import HORIZON
from scripts.render_lewm_action_prefix_a4_accuracy_gallery import (
    ALL_HORIZONS,
    _parser,
    _predict_window,
    _render_html,
)


def _metrics() -> dict[str, dict[str, dict[str, float | int]]]:
    rows = {
        f"H{horizon}": {
            "horizon": horizon,
            "sample_count": 2,
            "mean_epe_px": float(horizon),
            "median_epe_px": float(horizon),
            "p95_epe_px": float(horizon) + 0.5,
            "pck_at_4": 0.5,
            "pck_at_8": 1.0,
        }
        for horizon in ALL_HORIZONS
    }
    return {"decoder_floor": deepcopy(rows), "prediction": deepcopy(rows)}


class _Model(torch.nn.Module):
    def forward(
        self,
        current_state: torch.Tensor,
        future_action: torch.Tensor,
        task_index: torch.Tensor,
    ) -> torch.Tensor:
        assert current_state.shape == (1, 4, 192)
        assert future_action.shape == (1, HORIZON, 20)
        assert task_index.tolist() == [3]
        return current_state[:, None].expand(-1, HORIZON, -1, -1)


def test_predict_window_uses_latest_state_and_dense_action_prefix() -> None:
    history = np.zeros((3, 4, 192), dtype=np.float32)
    history[-1].fill(7.0)
    result = _predict_window(
        _Model(),
        history_embedding=history,
        future_action=np.zeros((HORIZON, 20), dtype=np.float32),
        task_index=3,
        device=torch.device("cpu"),
    )
    assert result.shape == (HORIZON, 4, 192)
    np.testing.assert_array_equal(result, np.full_like(result, 7.0))


def test_html_describes_nonrecursive_causal_contract() -> None:
    page = _render_html(
        [
            {
                "task": "push-v3",
                "task_label": "Push",
                "task_instance": 2,
                "environment_seed": 9,
                "decision_window_count": 4,
                "gif": 'bad"><script>x</script>.gif',
                "metrics": _metrics(),
            }
        ],
        {"seed": 40401, "step": 2000, "decoder": "decoder.npz"},
    )
    assert "never model inputs" in page
    assert "does not recursively feed predicted latent states" in page
    assert "Dense-prefix P95" in page
    assert "<script>x</script>" not in page
    assert "&lt;script&gt;x&lt;/script&gt;" in page


def test_cli_defaults_to_cuda_zero_and_rejects_nonpositive_windows() -> None:
    args = _parser().parse_args([])
    assert args.device == "cuda:0"
    with pytest.raises(SystemExit):
        _parser().parse_args(["--max-decision-windows", "0"])
