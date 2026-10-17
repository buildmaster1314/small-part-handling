"""Low-order camera-equivariant calibration for a frozen LeWM readout."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn

from fcpg.lewm_action_branch_a4 import HORIZON, SIGREG_WEIGHT, TASKS
from fcpg.lewm_action_prefix_joint_a4 import ANCHOR_COUNT


HARMONIC_PROTOCOL = "a4_orbit_harmonic_camera_readout_v1"
HARMONIC_CHECKPOINT_TYPE = "a4_orbit_harmonic_camera_readout"
HARMONIC_DIM = 5
INPUT_COMPONENT_DIM = 3


@dataclass(frozen=True)
class HarmonicReadoutOutput:
    predicted_q_px: torch.Tensor
    residual_q_px: torch.Tensor


def orbit_harmonic_features(clock_hour: torch.Tensor) -> torch.Tensor:
    if clock_hour.ndim != 1:
        raise ValueError("clock_hour must have shape [B]")
    if bool(torch.any((clock_hour < 1) | (clock_hour > 12))):
        raise ValueError("clock_hour must be in [1,12]")
    theta = torch.remainder(clock_hour.float(), 12.0) * (math.pi / 6.0)
    return torch.stack(
        [
            torch.ones_like(theta),
            torch.sin(theta),
            torch.cos(theta),
            torch.sin(2.0 * theta),
            torch.cos(2.0 * theta),
        ],
        dim=-1,
    )


class HarmonicCameraReadout(nn.Module):
    """Apply a smooth camera-angle correction to frozen LeWM displacement."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(
                len(TASKS),
                HORIZON,
                ANCHOR_COUNT,
                HARMONIC_DIM * INPUT_COMPONENT_DIM,
                2,
            ),
            requires_grad=False,
        )

    def forward(
        self,
        *,
        base_prediction_q_px: torch.Tensor,
        current_q_px: torch.Tensor,
        task_index: torch.Tensor,
        clock_hour: torch.Tensor,
        image_size: tuple[int, int],
    ) -> HarmonicReadoutOutput:
        batch_size = int(base_prediction_q_px.shape[0])
        if base_prediction_q_px.shape != (
            batch_size,
            HORIZON,
            ANCHOR_COUNT,
            2,
        ):
            raise ValueError("base prediction must have shape [B,8,3,2]")
        if current_q_px.shape != (batch_size, ANCHOR_COUNT, 2):
            raise ValueError("current_q_px must have shape [B,3,2]")
        if task_index.shape != (batch_size,) or clock_hour.shape != (batch_size,):
            raise ValueError("task_index and clock_hour must have shape [B]")
        if bool(torch.any((task_index < 0) | (task_index >= len(TASKS)))):
            raise ValueError("task_index is outside the seven-task panel")
        if not bool(torch.isfinite(base_prediction_q_px).all()) or not bool(
            torch.isfinite(current_q_px).all()
        ):
            raise ValueError("harmonic readout input contains non-finite values")
        width, height = (int(image_size[0]), int(image_size[1]))
        if width <= 1 or height <= 1:
            raise ValueError("image_size must be at least 2x2")
        scale = base_prediction_q_px.new_tensor([width - 1, height - 1]).float()
        displacement = base_prediction_q_px.float() - current_q_px.float()[:, None]
        augmented = torch.cat(
            [displacement / scale, torch.ones_like(displacement[..., :1])], dim=-1
        )
        harmonic = orbit_harmonic_features(clock_hour).to(
            device=displacement.device, dtype=displacement.dtype
        )
        design = (
            harmonic[:, None, None, :, None]
            * augmented[:, :, :, None, :]
        ).reshape(
            batch_size,
            HORIZON,
            ANCHOR_COUNT,
            HARMONIC_DIM * INPUT_COMPONENT_DIM,
        )
        selected_weight = self.weight[task_index.long()]
        residual = (
            torch.einsum("bhae,bhaeo->bhao", design, selected_weight) * scale
        )
        return HarmonicReadoutOutput(
            predicted_q_px=base_prediction_q_px.float() + residual,
            residual_q_px=residual,
        )


def harmonic_checkpoint(
    model: HarmonicCameraReadout,
    *,
    base_checkpoint: str,
    raw_dataset: str,
    ridge: float,
) -> dict[str, Any]:
    return {
        "checkpoint_type": HARMONIC_CHECKPOINT_TYPE,
        "protocol": HARMONIC_PROTOCOL,
        "base_checkpoint": str(base_checkpoint),
        "raw_dataset": str(raw_dataset),
        "base_sigreg_weight": SIGREG_WEIGHT,
        "ridge": float(ridge),
        "model_state_dict": model.state_dict(),
    }


def load_harmonic_checkpoint(checkpoint: Mapping[str, Any]) -> HarmonicCameraReadout:
    if checkpoint.get("checkpoint_type") != HARMONIC_CHECKPOINT_TYPE:
        raise ValueError("checkpoint is not a harmonic camera readout")
    if checkpoint.get("protocol") != HARMONIC_PROTOCOL:
        raise ValueError("harmonic camera readout protocol drifted")
    if float(checkpoint.get("base_sigreg_weight", -1.0)) != SIGREG_WEIGHT:
        raise ValueError("harmonic readout base SIGReg drifted")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("harmonic camera readout checkpoint is incomplete")
    model = HarmonicCameraReadout()
    model.load_state_dict(state, strict=True)
    return model


__all__ = [
    "HARMONIC_CHECKPOINT_TYPE",
    "HARMONIC_DIM",
    "HARMONIC_PROTOCOL",
    "HarmonicCameraReadout",
    "HarmonicReadoutOutput",
    "harmonic_checkpoint",
    "load_harmonic_checkpoint",
    "orbit_harmonic_features",
]
