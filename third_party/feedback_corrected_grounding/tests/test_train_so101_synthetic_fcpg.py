import numpy as np
import torch

from scripts.train_so101_synthetic_fcpg import masked_smooth_l1, metric_row


def test_masked_smooth_l1_ignores_invalid_anchors() -> None:
    prediction = torch.tensor([[[1.0, 0.0], [100.0, 100.0]]])
    target = torch.zeros_like(prediction)
    valid = torch.tensor([[True, False]])

    loss = masked_smooth_l1(prediction, target, valid)

    assert torch.isclose(loss, torch.tensor(0.25))


def test_metric_row_reports_pixel_improvement() -> None:
    nominal = np.asarray([[[0.0, 0.0], [10.0, 0.0]]], dtype=np.float32)
    target = np.asarray([[[4.0, 0.0], [12.0, 0.0]]], dtype=np.float32)
    corrected = np.asarray([[[3.0, 0.0], [12.0, 0.0]]], dtype=np.float32)

    result = metric_row(
        nominal,
        corrected,
        target,
        np.ones((1, 2), dtype=bool),
    )

    assert result["anchors"] == 2
    assert result["nominal_epe_px"] == 3.0
    assert result["corrected_epe_px"] == 0.5
