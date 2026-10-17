"""Non-recursive action-prefix dynamics for four-slot LeWM states.

The prefix encoder consumes one token derived from the current state followed
by the planned action tokens.  Causal attention makes the representation for
each horizon depend only on the state and the action prefix available at that
horizon.  A shared action-modulated residual MLP then predicts every horizon
directly from the same current state.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import torch
from torch import nn


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}
_DENSE_STATE_SLOTS = 4


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _optional_num_tasks(value: object) -> int | None:
    if value is None:
        return None
    return _positive_int(value, "num_tasks")


def _dropout_probability(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("dropout must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError("dropout must be finite and in [0, 1)")
    return result


def _sinusoidal_positions(length: int, dim: int) -> torch.Tensor:
    """Build standard fixed sinusoidal positions with shape ``[1, T, D]``."""

    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / float(dim))
    )
    encoding = torch.zeros(length, dim, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * frequency)
    encoding[:, 1::2] = torch.cos(
        position * frequency[: encoding[:, 1::2].shape[1]]
    )
    return encoding.unsqueeze(0)


class _ActionModulatedResidualBlock(nn.Module):
    """AdaLN-zero residual MLP conditioned by one action-prefix token."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        fusion_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, 3 * dim),
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self,
        value: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale, gate = self.modulation(condition).chunk(3, dim=-1)
        shift = shift.unsqueeze(-2)
        scale = scale.unsqueeze(-2)
        gate = gate.unsqueeze(-2)
        modulated = self.norm(value) * (1.0 + scale) + shift
        return value + gate * self.mlp(modulated)


class _ActionModulatedResidualMLP(nn.Module):
    """Predict per-slot deltas while sharing parameters across all horizons."""

    def __init__(
        self,
        dim: int,
        depth: int,
        hidden_dim: int,
        fusion_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _ActionModulatedResidualBlock(
                    dim,
                    hidden_dim,
                    fusion_dim,
                    dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        current_state: torch.Tensor,
        prefix_condition: torch.Tensor,
    ) -> torch.Tensor:
        value = current_state
        for layer in self.layers:
            value = layer(value, prefix_condition)
        return value - current_state


class ActionPrefixDynamics(nn.Module):
    """Fast-LeWM-style dense dynamics over a future action prefix.

    Args:
        action_dim: Dimension ``A`` of each future action.
        state_dim: Dimension ``D`` of each state slot.
        num_slots: Number ``S`` of state slots. The intended schema has four
            slots: one context slot followed by three persistent anchors.
        num_tasks: Number of task identities, or ``None`` to disable task
            conditioning. ``task_index`` remains optional when configured.
        max_horizon: Maximum accepted action-prefix length.
        prefix_depth: Number of causal Transformer encoder layers.
        prefix_heads: Number of attention heads in each prefix layer.
        predictor_depth: Number of shared residual MLP layers.
        dropout: Dropout used by both the prefix encoder and predictor.

    Input shapes are ``current_state=[B,S,D]`` and
    ``future_actions=[B,H,A]``. The output has shape ``[B,H,S,D]``. Every
    horizon starts from ``current_state``; predicted horizons are never fed
    back into the model.
    """

    def __init__(
        self,
        action_dim: int,
        state_dim: int = 192,
        num_slots: int = 4,
        num_tasks: int | None = None,
        max_horizon: int = 8,
        prefix_depth: int = 3,
        prefix_heads: int = 6,
        predictor_depth: int = 6,
        dropout: float = 0.1,
        prefix_mlp_dim: int | None = None,
        predictor_hidden_dim: int | None = None,
        fusion_dim: int = 768,
        state_mlp_dim: int = 768,
    ) -> None:
        super().__init__()
        self.state_dim = _positive_int(state_dim, "state_dim")
        self.action_dim = _positive_int(action_dim, "action_dim")
        self.num_slots = _positive_int(num_slots, "num_slots")
        self.num_tasks = _optional_num_tasks(num_tasks)
        self.max_horizon = _positive_int(max_horizon, "max_horizon")
        self.prefix_depth = _positive_int(prefix_depth, "prefix_depth")
        self.prefix_heads = _positive_int(prefix_heads, "prefix_heads")
        self.predictor_depth = _positive_int(predictor_depth, "predictor_depth")
        self.dropout = _dropout_probability(dropout)
        self.fusion_dim = _positive_int(fusion_dim, "fusion_dim")
        self.state_mlp_dim = _positive_int(state_mlp_dim, "state_mlp_dim")
        if self.state_dim % self.prefix_heads != 0:
            raise ValueError(
                f"state_dim={self.state_dim} must be divisible by "
                f"prefix_heads={self.prefix_heads}"
            )

        prefix_hidden = (
            4 * self.state_dim
            if prefix_mlp_dim is None
            else _positive_int(prefix_mlp_dim, "prefix_mlp_dim")
        )
        predictor_hidden = (
            2048
            if predictor_hidden_dim is None
            else _positive_int(predictor_hidden_dim, "predictor_hidden_dim")
        )

        self.state_projection = nn.Sequential(
            nn.LayerNorm(self.num_slots * self.state_dim),
            nn.Linear(self.num_slots * self.state_dim, self.state_mlp_dim),
            nn.GELU(),
            nn.Linear(self.state_mlp_dim, self.state_dim),
        )
        self.action_projection = nn.Sequential(
            nn.Linear(self.action_dim, self.state_dim),
            nn.GELU(),
            nn.Linear(self.state_dim, self.state_dim),
        )
        self.task_embedding = (
            None
            if self.num_tasks is None
            else nn.Embedding(self.num_tasks, self.state_dim)
        )

        prefix_layer = nn.TransformerEncoderLayer(
            d_model=self.state_dim,
            nhead=self.prefix_heads,
            dim_feedforward=prefix_hidden,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.prefix_encoder = nn.TransformerEncoder(
            prefix_layer,
            num_layers=self.prefix_depth,
            norm=nn.LayerNorm(self.state_dim),
            enable_nested_tensor=False,
        )
        self.residual_predictor = _ActionModulatedResidualMLP(
            dim=self.state_dim,
            depth=self.predictor_depth,
            hidden_dim=predictor_hidden,
            fusion_dim=self.fusion_dim,
            dropout=self.dropout,
        )

        self.register_buffer(
            "action_mean",
            torch.zeros(self.action_dim, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "action_std",
            torch.ones(self.action_dim, dtype=torch.float32),
            persistent=True,
        )

        sequence_length = self.max_horizon + 1
        self.register_buffer(
            "position_encoding",
            _sinusoidal_positions(sequence_length, self.state_dim),
            persistent=False,
        )
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(sequence_length, sequence_length, dtype=torch.bool),
                diagonal=1,
            ),
            persistent=False,
        )

    def set_action_normalization(
        self,
        mean: torch.Tensor | list[float],
        std: torch.Tensor | list[float],
    ) -> None:
        """Install finite train-split action statistics."""

        mean_tensor = torch.as_tensor(
            mean,
            dtype=self.action_mean.dtype,
            device=self.action_mean.device,
        )
        std_tensor = torch.as_tensor(
            std,
            dtype=self.action_std.dtype,
            device=self.action_std.device,
        )
        expected = (self.action_dim,)
        if mean_tensor.shape != expected or std_tensor.shape != expected:
            raise ValueError(
                f"action normalization must have shape {expected}, got "
                f"{tuple(mean_tensor.shape)} and {tuple(std_tensor.shape)}"
            )
        if not bool(torch.isfinite(mean_tensor).all()) or not bool(
            torch.isfinite(std_tensor).all()
        ):
            raise ValueError("action normalization must be finite")
        if bool(torch.any(std_tensor <= 0.0)):
            raise ValueError("action_std must be strictly positive")
        with torch.no_grad():
            self.action_mean.copy_(mean_tensor)
            self.action_std.copy_(std_tensor)

    def _validated_inputs(
        self,
        current_state: torch.Tensor,
        future_actions: torch.Tensor,
        task_index: torch.Tensor | None,
    ) -> tuple[int, int, torch.Tensor | None]:
        if not torch.is_tensor(current_state):
            raise TypeError("current_state must be a tensor")
        if not torch.is_tensor(future_actions):
            raise TypeError("future_actions must be a tensor")
        if current_state.ndim != 3 or current_state.shape[1:] != (
            self.num_slots,
            self.state_dim,
        ):
            raise ValueError(
                "current_state must have shape "
                f"[B, {self.num_slots}, {self.state_dim}], "
                f"got {tuple(current_state.shape)}"
            )
        batch_size = int(current_state.shape[0])
        if batch_size <= 0:
            raise ValueError("current_state batch dimension must be positive")
        if future_actions.ndim != 3 or future_actions.shape[0] != batch_size:
            raise ValueError(
                "future_actions must have shape "
                f"[B, H, {self.action_dim}], got {tuple(future_actions.shape)}"
            )
        if future_actions.shape[2] != self.action_dim:
            raise ValueError(
                "future_actions must have shape "
                f"[B, H, {self.action_dim}], got {tuple(future_actions.shape)}"
            )
        horizon = int(future_actions.shape[1])
        if not 0 < horizon <= self.max_horizon:
            raise ValueError(
                f"future action horizon must be in [1, {self.max_horizon}], "
                f"got {horizon}"
            )
        if not current_state.is_floating_point():
            raise TypeError("current_state must be floating point")
        if not future_actions.is_floating_point():
            raise TypeError("future_actions must be floating point")
        if current_state.dtype != future_actions.dtype:
            raise TypeError("current_state and future_actions must share dtype")
        if current_state.device != future_actions.device:
            raise ValueError("current_state and future_actions must share device")
        if not bool(torch.isfinite(current_state).all()):
            raise ValueError("current_state contains non-finite values")
        if not bool(torch.isfinite(future_actions).all()):
            raise ValueError("future_actions contains non-finite values")

        validated_task: torch.Tensor | None = None
        if task_index is not None:
            if self.task_embedding is None or self.num_tasks is None:
                raise ValueError(
                    "task_index requires ActionPrefixDynamics(num_tasks=...)"
                )
            if not torch.is_tensor(task_index):
                raise TypeError("task_index must be a tensor")
            if task_index.shape != (batch_size,):
                raise ValueError(
                    f"task_index must have shape {(batch_size,)}, "
                    f"got {tuple(task_index.shape)}"
                )
            if task_index.dtype not in _INTEGER_DTYPES:
                raise TypeError("task_index must have an integer dtype")
            minimum = int(task_index.detach().amin().item())
            maximum = int(task_index.detach().amax().item())
            if minimum < 0 or maximum >= self.num_tasks:
                raise ValueError(
                    f"task_index values must be in [0, {self.num_tasks - 1}], "
                    f"got range [{minimum}, {maximum}]"
                )
            validated_task = task_index.to(
                device=current_state.device,
                dtype=torch.long,
            )
        return batch_size, horizon, validated_task

    def _encode_validated_prefix(
        self,
        current_state: torch.Tensor,
        future_actions: torch.Tensor,
        task_index: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, horizon = future_actions.shape[:2]
        state_token = self.state_projection(
            current_state.reshape(batch_size, self.num_slots * self.state_dim)
        )
        if task_index is not None:
            assert self.task_embedding is not None
            state_token = state_token + self.task_embedding(task_index)
        normalized_actions = (
            future_actions - self.action_mean.to(dtype=future_actions.dtype)
        ) / self.action_std.to(dtype=future_actions.dtype)
        action_tokens = self.action_projection(normalized_actions)
        tokens = torch.cat([state_token.unsqueeze(1), action_tokens], dim=1)
        tokens = tokens + self.position_encoding[
            :, : horizon + 1
        ].to(dtype=tokens.dtype)
        encoded = self.prefix_encoder(
            tokens,
            mask=self.causal_mask[: horizon + 1, : horizon + 1],
        )
        return encoded[:, 1:]

    def encode_prefix(
        self,
        current_state: torch.Tensor,
        future_actions: torch.Tensor,
        task_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return causal per-horizon prefix conditions with shape ``[B,H,D]``."""

        _, _, validated_task = self._validated_inputs(
            current_state,
            future_actions,
            task_index,
        )
        return self._encode_validated_prefix(
            current_state,
            future_actions,
            validated_task,
        )

    def forward(
        self,
        current_state: torch.Tensor,
        future_actions: torch.Tensor,
        task_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, horizon, validated_task = self._validated_inputs(
            current_state,
            future_actions,
            task_index,
        )
        prefix_condition = self._encode_validated_prefix(
            current_state,
            future_actions,
            validated_task,
        )
        shared_current = current_state[:, None].expand(
            batch_size,
            horizon,
            self.num_slots,
            self.state_dim,
        )
        residual = self.residual_predictor(shared_current, prefix_condition)
        return shared_current + residual


def _loss_weight(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def dense_prefix_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    context_weight: float = 0.25,
    anchor_weight: float = 0.75,
    horizon_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted dense-prefix MSE for one context and three anchor slots.

    ``horizon_weights`` is optional so the historical uniform objective stays
    compatible. When supplied, it is normalized to mean one and applies only
    across prediction horizons, not across physical slots.
    """

    if not torch.is_tensor(pred) or not torch.is_tensor(target):
        raise TypeError("pred and target must be tensors")
    if pred.ndim != 4 or pred.shape[2] != _DENSE_STATE_SLOTS:
        raise ValueError("pred must have shape [B, H, 4, D]")
    if pred.shape != target.shape:
        raise ValueError(
            "pred and target must share shape [B, H, 4, D], got "
            f"{tuple(pred.shape)} and {tuple(target.shape)}"
        )
    if any(int(size) <= 0 for size in pred.shape):
        raise ValueError("pred and target dimensions must be positive")
    if not pred.is_floating_point() or not target.is_floating_point():
        raise TypeError("pred and target must be floating point")
    if pred.dtype != target.dtype:
        raise TypeError("pred and target must share dtype")
    if pred.device != target.device:
        raise ValueError("pred and target must share device")
    if not bool(torch.isfinite(pred).all()):
        raise ValueError("pred contains non-finite values")
    if not bool(torch.isfinite(target).all()):
        raise ValueError("target contains non-finite values")

    resolved_context_weight = _loss_weight(context_weight, "context_weight")
    resolved_anchor_weight = _loss_weight(anchor_weight, "anchor_weight")
    if resolved_context_weight == 0.0 and resolved_anchor_weight == 0.0:
        raise ValueError("at least one loss weight must be positive")

    difference = pred - target
    if difference.dtype in {torch.float16, torch.bfloat16}:
        difference = difference.float()
    if horizon_weights is None:
        resolved_horizon_weights = torch.ones(
            pred.shape[1], device=difference.device, dtype=difference.dtype
        )
    else:
        if not torch.is_tensor(horizon_weights):
            raise TypeError("horizon_weights must be a tensor or None")
        if horizon_weights.ndim != 1 or horizon_weights.shape[0] != pred.shape[1]:
            raise ValueError("horizon_weights must have shape [H]")
        if not horizon_weights.is_floating_point():
            raise TypeError("horizon_weights must be floating point")
        if horizon_weights.device != pred.device:
            raise ValueError("horizon_weights must share the prediction device")
        if not bool(torch.isfinite(horizon_weights).all()):
            raise ValueError("horizon_weights contains non-finite values")
        if bool(torch.any(horizon_weights < 0.0)):
            raise ValueError("horizon_weights must be non-negative")
        if float(horizon_weights.sum()) <= 0.0:
            raise ValueError("horizon_weights must contain a positive value")
        resolved_horizon_weights = horizon_weights.to(difference.dtype)
        resolved_horizon_weights = resolved_horizon_weights / resolved_horizon_weights.mean()
    context_per_horizon = difference[:, :, 0].square().mean(dim=-1)
    anchor_per_horizon = difference[:, :, 1:].square().mean(dim=(-1, -2))
    context_mse = (context_per_horizon * resolved_horizon_weights[None]).mean()
    anchor_mse = (anchor_per_horizon * resolved_horizon_weights[None]).mean()
    loss = (
        resolved_context_weight * context_mse
        + resolved_anchor_weight * anchor_mse
    )
    if not bool(torch.isfinite(loss.detach())):
        raise ValueError("dense prefix MSE is non-finite")
    return loss


__all__ = ["ActionPrefixDynamics", "dense_prefix_mse"]
