"""Bounded anchor-aware adapters for the seven-task camera-action LeWM."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from fcpg.lewm_action_branch_a4 import HORIZON, SIGREG_WEIGHT, TASKS
from fcpg.lewm_action_prefix import ActionPrefixDynamics
from fcpg.lewm_action_prefix_joint_a4 import LATENT_DIM, STATE_SLOTS
from fcpg.lewm_camera_action_joint_a4 import (
    CameraActionJointSevenTaskLeWM,
    load_camera_action_joint_checkpoint,
)


ADAPTER_PROTOCOL = "a4_camera_action_bounded_anchor_adapter_v1"
ADAPTER_CHECKPOINT_TYPE = "a4_camera_action_bounded_anchor_adapter"
ADAPTER_VARIANTS = ("transformer", "mlp")
DEFAULT_ADAPTER_CONFIG = {
    "model_dim": 192,
    "depth": 2,
    "heads": 6,
    "ffn_dim": 384,
    "dropout": 0.1,
}


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _dropout(value: object) -> float:
    result = float(value)
    if not 0.0 <= result < 1.0:
        raise ValueError("dropout must be in [0,1)")
    return result


def _variant(value: object) -> str:
    result = str(value)
    if result not in ADAPTER_VARIANTS:
        raise ValueError(
            f"adapter variant must be one of {ADAPTER_VARIANTS}, got {result!r}"
        )
    return result


class _PerTokenResidualBlock(nn.Module):
    """Parameter-matched token-local alternative to cross-slot attention."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.fc2(
            self.dropout(torch.nn.functional.gelu(self.fc1(self.norm(value))))
        )
        return value + self.dropout(residual)


class AnchorResidualAdapter(nn.Module):
    """Predict a latent residual from base dynamics and persistent slot tokens."""

    def __init__(
        self,
        *,
        variant: str,
        state_dim: int = LATENT_DIM,
        num_slots: int = STATE_SLOTS,
        num_tasks: int = len(TASKS),
        max_horizon: int = HORIZON,
        model_dim: int = 192,
        depth: int = 2,
        heads: int = 6,
        ffn_dim: int = 384,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.variant = _variant(variant)
        self.state_dim = _positive_int(state_dim, "state_dim")
        self.num_slots = _positive_int(num_slots, "num_slots")
        self.num_tasks = _positive_int(num_tasks, "num_tasks")
        self.max_horizon = _positive_int(max_horizon, "max_horizon")
        self.model_dim = _positive_int(model_dim, "model_dim")
        self.depth = _positive_int(depth, "depth")
        self.heads = _positive_int(heads, "heads")
        self.ffn_dim = _positive_int(ffn_dim, "ffn_dim")
        self.dropout_probability = _dropout(dropout)
        if self.model_dim % self.heads != 0:
            raise ValueError("model_dim must be divisible by heads")

        self.input_projection = nn.Linear(4 * self.state_dim, self.model_dim)
        self.slot_embedding = nn.Embedding(self.num_slots, self.model_dim)
        self.horizon_embedding = nn.Embedding(self.max_horizon, self.model_dim)
        self.task_embedding = nn.Embedding(self.num_tasks, self.model_dim)

        if self.variant == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=self.model_dim,
                nhead=self.heads,
                dim_feedforward=self.ffn_dim,
                dropout=self.dropout_probability,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.mixer: nn.Module = nn.TransformerEncoder(
                layer,
                num_layers=self.depth,
                enable_nested_tensor=False,
            )
        else:
            # Match the Transformer layer's attention + FFN parameter budget.
            local_hidden_dim = 2 * self.ffn_dim
            self.mixer = nn.Sequential(
                *[
                    _PerTokenResidualBlock(
                        self.model_dim,
                        local_hidden_dim,
                        self.dropout_probability,
                    )
                    for _ in range(self.depth)
                ]
            )
        self.output_norm = nn.LayerNorm(self.model_dim)
        self.output_head = nn.Linear(self.model_dim, self.state_dim)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def config(self) -> dict[str, Any]:
        return {
            "model_dim": self.model_dim,
            "depth": self.depth,
            "heads": self.heads,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout_probability,
        }

    def _validate(
        self,
        current_state: torch.Tensor,
        base_prediction: torch.Tensor,
        prefix_condition: torch.Tensor,
        task_index: torch.Tensor,
    ) -> tuple[int, int]:
        if current_state.ndim != 3 or current_state.shape[1:] != (
            self.num_slots,
            self.state_dim,
        ):
            raise ValueError("current_state has an invalid adapter shape")
        batch_size = int(current_state.shape[0])
        if base_prediction.ndim != 4 or base_prediction.shape[0] != batch_size:
            raise ValueError("base_prediction must have shape [B,H,S,D]")
        horizon = int(base_prediction.shape[1])
        if base_prediction.shape[2:] != (self.num_slots, self.state_dim):
            raise ValueError("base_prediction has an invalid slot shape")
        if not 0 < horizon <= self.max_horizon:
            raise ValueError("adapter horizon is outside its configured range")
        if prefix_condition.shape != (batch_size, horizon, self.state_dim):
            raise ValueError("prefix_condition must have shape [B,H,D]")
        if task_index.shape != (batch_size,):
            raise ValueError("task_index must have shape [B]")
        if task_index.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("task_index must be integral")
        if (
            int(task_index.amin().item()) < 0
            or int(task_index.amax().item()) >= self.num_tasks
        ):
            raise ValueError("task_index is outside the configured task range")
        tensors = (current_state, base_prediction, prefix_condition)
        if any(not value.is_floating_point() for value in tensors):
            raise TypeError("adapter states must be floating point")
        if len({value.device for value in tensors}) != 1:
            raise ValueError("adapter states must share device")
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("adapter states contain non-finite values")
        return batch_size, horizon

    def forward(
        self,
        current_state: torch.Tensor,
        base_prediction: torch.Tensor,
        prefix_condition: torch.Tensor,
        task_index: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, horizon = self._validate(
            current_state,
            base_prediction,
            prefix_condition,
            task_index,
        )
        compute_dtype = base_prediction.dtype
        current = current_state.to(dtype=compute_dtype)[:, None].expand(
            -1, horizon, -1, -1
        )
        prefix = prefix_condition.to(dtype=compute_dtype)[:, :, None].expand(
            -1, -1, self.num_slots, -1
        )
        features = torch.cat(
            [current, base_prediction, base_prediction - current, prefix],
            dim=-1,
        )
        tokens = self.input_projection(features)
        slot_ids = torch.arange(self.num_slots, device=tokens.device)
        horizon_ids = torch.arange(horizon, device=tokens.device)
        tokens = (
            tokens
            + self.slot_embedding(slot_ids)[None, None]
            + self.horizon_embedding(horizon_ids)[None, :, None]
            + self.task_embedding(task_index.long())[:, None, None]
        )
        mixed = self.mixer(
            tokens.reshape(batch_size * horizon, self.num_slots, self.model_dim)
        )
        raw = self.output_head(self.output_norm(mixed))
        return raw.reshape(
            batch_size,
            horizon,
            self.num_slots,
            self.state_dim,
        )


class BoundedAnchorAdapterDynamics(nn.Module):
    """Add a strictly bounded adapter residual to a frozen proven dynamics."""

    def __init__(
        self,
        base_dynamics: ActionPrefixDynamics,
        adapter: AnchorResidualAdapter,
        residual_cap: torch.Tensor,
    ) -> None:
        super().__init__()
        if not isinstance(base_dynamics, ActionPrefixDynamics):
            raise TypeError("base_dynamics must be ActionPrefixDynamics")
        if not isinstance(adapter, AnchorResidualAdapter):
            raise TypeError("adapter must be AnchorResidualAdapter")
        expected = (
            base_dynamics.max_horizon,
            base_dynamics.num_slots,
            base_dynamics.state_dim,
        )
        cap = torch.as_tensor(residual_cap, dtype=torch.float32)
        if cap.shape != expected:
            raise ValueError(f"residual_cap must have shape {expected}")
        if not bool(torch.isfinite(cap).all()) or bool(torch.any(cap <= 0.0)):
            raise ValueError("residual_cap must be finite and strictly positive")
        if (
            adapter.state_dim != base_dynamics.state_dim
            or adapter.num_slots != base_dynamics.num_slots
            or adapter.max_horizon != base_dynamics.max_horizon
            or adapter.num_tasks != base_dynamics.num_tasks
        ):
            raise ValueError("adapter and base dynamics contracts differ")
        self.base_dynamics = base_dynamics
        self.adapter = adapter
        self.register_buffer("residual_cap", cap, persistent=True)

    @property
    def action_dim(self) -> int:
        return int(self.base_dynamics.action_dim)

    @property
    def state_dim(self) -> int:
        return int(self.base_dynamics.state_dim)

    @property
    def num_slots(self) -> int:
        return int(self.base_dynamics.num_slots)

    @property
    def max_horizon(self) -> int:
        return int(self.base_dynamics.max_horizon)

    @property
    def num_tasks(self) -> int:
        if self.base_dynamics.num_tasks is None:
            raise RuntimeError("base dynamics does not expose a task contract")
        return int(self.base_dynamics.num_tasks)

    def set_action_normalization(
        self,
        mean: torch.Tensor | list[float],
        std: torch.Tensor | list[float],
    ) -> None:
        self.base_dynamics.set_action_normalization(mean, std)

    def bounded_delta(self, raw_delta: torch.Tensor) -> torch.Tensor:
        if raw_delta.ndim != 4 or raw_delta.shape[2:] != (
            self.num_slots,
            self.state_dim,
        ):
            raise ValueError("raw_delta must have shape [B,H,S,D]")
        horizon = int(raw_delta.shape[1])
        if not 0 < horizon <= self.max_horizon:
            raise ValueError("raw_delta horizon is outside the configured range")
        cap = self.residual_cap[:horizon].to(
            device=raw_delta.device,
            dtype=raw_delta.dtype,
        )
        return cap[None] * torch.tanh(raw_delta / cap[None])

    def forward(
        self,
        current_state: torch.Tensor,
        future_actions: torch.Tensor,
        task_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if task_index is None:
            raise ValueError("bounded anchor adapter requires task_index")
        prefix = self.base_dynamics.encode_prefix(
            current_state,
            future_actions,
            task_index=task_index,
        )
        batch_size, horizon = prefix.shape[:2]
        current = current_state[:, None].expand(
            batch_size,
            horizon,
            self.num_slots,
            self.state_dim,
        )
        base_prediction = current + self.base_dynamics.residual_predictor(
            current,
            prefix,
        )
        raw_delta = self.adapter(
            current_state,
            base_prediction,
            prefix,
            task_index,
        )
        return base_prediction + self.bounded_delta(raw_delta)


def _adapter_from_config(
    *,
    variant: str,
    base_dynamics: ActionPrefixDynamics,
    config: Mapping[str, Any],
) -> AnchorResidualAdapter:
    if base_dynamics.num_tasks is None:
        raise ValueError("camera-action parent dynamics must be task conditioned")
    return AnchorResidualAdapter(
        variant=variant,
        state_dim=base_dynamics.state_dim,
        num_slots=base_dynamics.num_slots,
        num_tasks=int(base_dynamics.num_tasks),
        max_horizon=base_dynamics.max_horizon,
        model_dim=int(config["model_dim"]),
        depth=int(config["depth"]),
        heads=int(config["heads"]),
        ffn_dim=int(config["ffn_dim"]),
        dropout=float(config["dropout"]),
    )


def build_camera_action_adapter_model(
    parent_checkpoint: Mapping[str, Any],
    *,
    variant: str,
    residual_cap: torch.Tensor,
    seed: int,
    adapter_config: Mapping[str, Any] = DEFAULT_ADAPTER_CONFIG,
) -> CameraActionJointSevenTaskLeWM:
    """Freeze a camera-action parent and attach one trainable adapter."""

    resolved_variant = _variant(variant)
    model = load_camera_action_joint_checkpoint(parent_checkpoint)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    base_dynamics = model.base_model.dynamics
    if not isinstance(base_dynamics, ActionPrefixDynamics):
        raise TypeError("camera-action parent dynamics contract drifted")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        adapter = _adapter_from_config(
            variant=resolved_variant,
            base_dynamics=base_dynamics,
            config=adapter_config,
        )
    model.base_model.dynamics = BoundedAnchorAdapterDynamics(
        base_dynamics,
        adapter,
        residual_cap,
    )
    return model


def adapter_checkpoint_payload(
    model: CameraActionJointSevenTaskLeWM,
    *,
    step: int,
    seed: int,
    parent_checkpoint: str,
    raw_dataset: str,
) -> dict[str, Any]:
    dynamics = model.dynamics
    if not isinstance(dynamics, BoundedAnchorAdapterDynamics):
        raise TypeError("adapter checkpoint requires bounded adapter dynamics")
    return {
        "checkpoint_type": ADAPTER_CHECKPOINT_TYPE,
        "protocol": ADAPTER_PROTOCOL,
        "step": int(step),
        "seed": int(seed),
        "tasks": list(TASKS),
        "sigreg_weight": SIGREG_WEIGHT,
        "sigreg_gradient_active": False,
        "variant": dynamics.adapter.variant,
        "adapter_config": dynamics.adapter.config(),
        "parent_checkpoint": str(parent_checkpoint),
        "raw_dataset": str(raw_dataset),
        "residual_cap": dynamics.residual_cap.detach().cpu(),
        "adapter_state": dynamics.adapter.state_dict(),
    }


def load_camera_action_adapter_checkpoint(
    checkpoint: Mapping[str, Any],
) -> CameraActionJointSevenTaskLeWM:
    if checkpoint.get("checkpoint_type") != ADAPTER_CHECKPOINT_TYPE:
        raise ValueError("checkpoint is not a bounded camera-action adapter")
    if checkpoint.get("protocol") != ADAPTER_PROTOCOL:
        raise ValueError("camera-action adapter protocol drifted")
    if tuple(checkpoint.get("tasks", ())) != TASKS:
        raise ValueError("camera-action adapter task panel drifted")
    if float(checkpoint.get("sigreg_weight", -1.0)) != SIGREG_WEIGHT:
        raise ValueError("camera-action adapter SIGReg must be exactly 0.02")
    if checkpoint.get("sigreg_gradient_active") is not False:
        raise ValueError("frozen adapter checkpoint has an invalid SIGReg contract")
    parent_path = str(checkpoint.get("parent_checkpoint", ""))
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if not isinstance(parent, Mapping):
        raise TypeError("camera-action adapter parent must contain a mapping")
    cap = checkpoint.get("residual_cap")
    config = checkpoint.get("adapter_config")
    state = checkpoint.get("adapter_state")
    if not torch.is_tensor(cap) or not isinstance(config, Mapping) or not isinstance(
        state, Mapping
    ):
        raise ValueError("camera-action adapter checkpoint is incomplete")
    model = build_camera_action_adapter_model(
        parent,
        variant=str(checkpoint.get("variant", "")),
        residual_cap=cap,
        seed=int(checkpoint.get("seed", 0)),
        adapter_config=config,
    )
    dynamics = model.dynamics
    assert isinstance(dynamics, BoundedAnchorAdapterDynamics)
    dynamics.adapter.load_state_dict(state, strict=True)
    return model


__all__ = [
    "ADAPTER_CHECKPOINT_TYPE",
    "ADAPTER_PROTOCOL",
    "ADAPTER_VARIANTS",
    "DEFAULT_ADAPTER_CONFIG",
    "AnchorResidualAdapter",
    "BoundedAnchorAdapterDynamics",
    "adapter_checkpoint_payload",
    "build_camera_action_adapter_model",
    "load_camera_action_adapter_checkpoint",
]
