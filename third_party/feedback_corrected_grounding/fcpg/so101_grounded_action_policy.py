"""Grounded SO-101 action policy used by the real-task pilot checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class SO101GroundedActionOutput:
    """Absolute normalized commands and LeWM-compatible action chunks."""

    action: torch.Tensor
    delta: torch.Tensor
    action_chunks: torch.Tensor
    latent: torch.Tensor


class SO101GroundedActionPolicy(nn.Module):
    """Predict 40 absolute SO-101 commands from causal grounded history."""

    def __init__(
        self,
        *,
        latent_dim: int = 192,
        state_dim: int = 6,
        num_views: int = 2,
        num_anchors: int = 3,
        obs_horizon: int = 3,
        action_horizon: int = 40,
        raw_steps_per_chunk: int = 5,
        model_dim: int = 192,
        encoder_layers: int = 2,
        decoder_layers: int = 4,
        heads: int = 6,
        ffn_dim: int = 768,
        dropout: float = 0.0,
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        if state_dim != 6:
            raise ValueError("SO-101 action policy requires six joint positions")
        if latent_dim != model_dim:
            raise ValueError("the released checkpoint requires latent_dim == model_dim")
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        self.latent_dim = int(latent_dim)
        self.state_dim = int(state_dim)
        self.num_views = int(num_views)
        self.num_anchors = int(num_anchors)
        self.obs_horizon = int(obs_horizon)
        self.action_horizon = int(action_horizon)
        self.raw_steps_per_chunk = int(raw_steps_per_chunk)
        self.model_dim = int(model_dim)

        self.frame_embedding = nn.Parameter(torch.empty(obs_horizon, model_dim))
        self.view_embedding = nn.Parameter(torch.empty(num_views, model_dim))
        self.anchor_embedding = nn.Parameter(torch.empty(num_anchors, model_dim))
        self.global_type_embedding = nn.Parameter(torch.empty(model_dim))
        self.state_type_embedding = nn.Parameter(torch.empty(model_dim))
        self.past_action_type_embedding = nn.Parameter(torch.empty(model_dim))
        self.action_queries = nn.Parameter(torch.empty(1, action_horizon, model_dim))
        self.action_position = nn.Parameter(torch.empty(1, action_horizon, model_dim))

        self.register_buffer("step_cap", torch.ones(state_dim))
        self.register_buffer("action_lower", -torch.ones(state_dim))
        self.register_buffer("action_upper", torch.ones(state_dim))

        self.context_projection = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, model_dim)
        )
        self.anchor_projection = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, model_dim)
        )
        self.state_projection = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.past_action_projection = nn.Sequential(
            nn.LayerNorm(raw_steps_per_chunk * state_dim),
            nn.Linear(raw_steps_per_chunk * state_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        self.memory_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layers,
            norm=nn.LayerNorm(model_dim),
        )
        self.action_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers,
            norm=nn.LayerNorm(model_dim),
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, state_dim),
        )
        nn.init.normal_(self.frame_embedding, std=0.02)
        nn.init.normal_(self.view_embedding, std=0.02)
        nn.init.normal_(self.anchor_embedding, std=0.02)
        nn.init.normal_(self.global_type_embedding, std=0.02)
        nn.init.normal_(self.state_type_embedding, std=0.02)
        nn.init.normal_(self.past_action_type_embedding, std=0.02)
        nn.init.normal_(self.action_queries, std=0.02)
        nn.init.normal_(self.action_position, std=0.02)
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def set_action_contract(
        self,
        step_cap: torch.Tensor | np.ndarray,
        lower: torch.Tensor | np.ndarray,
        upper: torch.Tensor | np.ndarray,
    ) -> None:
        """Install normalized per-step and absolute command bounds."""

        cap = torch.as_tensor(step_cap, dtype=self.step_cap.dtype)
        lower = torch.as_tensor(lower, dtype=self.action_lower.dtype)
        upper = torch.as_tensor(upper, dtype=self.action_upper.dtype)
        expected = (self.state_dim,)
        if tuple(cap.shape) != expected:
            raise ValueError(f"step_cap must have shape {expected}")
        if tuple(lower.shape) != expected or tuple(upper.shape) != expected:
            raise ValueError(f"action bounds must have shape {expected}")
        if torch.any(cap <= 0) or torch.any(lower >= upper):
            raise ValueError("action contract is invalid")
        self.step_cap.copy_(cap)
        self.action_lower.copy_(lower)
        self.action_upper.copy_(upper)

    def forward(
        self,
        per_view_context: torch.Tensor,
        per_view_anchor_tokens: torch.Tensor,
        anchor_valid: torch.Tensor,
        state_history: torch.Tensor,
        past_action_chunks: torch.Tensor,
    ) -> SO101GroundedActionOutput:
        batch_size = state_history.shape[0]
        expected_context = (
            batch_size,
            self.obs_horizon,
            self.num_views,
            self.latent_dim,
        )
        expected_anchors = expected_context[:-1] + (
            self.num_anchors,
            self.latent_dim,
        )
        if tuple(per_view_context.shape) != expected_context:
            raise ValueError(
                f"per_view_context must have shape {expected_context}, "
                f"got {tuple(per_view_context.shape)}"
            )
        if tuple(per_view_anchor_tokens.shape) != expected_anchors:
            raise ValueError(
                f"per_view_anchor_tokens must have shape {expected_anchors}, "
                f"got {tuple(per_view_anchor_tokens.shape)}"
            )
        if tuple(anchor_valid.shape) != expected_anchors[:-1]:
            raise ValueError(
                f"anchor_valid must have shape {expected_anchors[:-1]}, "
                f"got {tuple(anchor_valid.shape)}"
            )
        if tuple(state_history.shape) != (
            batch_size,
            self.obs_horizon,
            self.state_dim,
        ):
            raise ValueError("state_history has an invalid shape")
        if tuple(past_action_chunks.shape) != (
            batch_size,
            self.obs_horizon - 1,
            self.raw_steps_per_chunk * self.state_dim,
        ):
            raise ValueError(
                "past_action_chunks must contain one flattened action chunk "
                "between each pair of observations"
            )

        frame = self.frame_embedding[None, :, None, :]
        view = self.view_embedding[None, None, :, :]
        context = (
            self.context_projection(per_view_context)
            + frame
            + view
            + self.global_type_embedding
        )
        finite_anchors = torch.where(
            anchor_valid[..., None],
            per_view_anchor_tokens,
            torch.zeros_like(per_view_anchor_tokens),
        )
        anchors = (
            self.anchor_projection(finite_anchors)
            + frame[:, :, :, None, :]
            + view[:, :, :, None, :]
            + self.anchor_embedding[None, None, None, :, :]
        )
        state = (
            self.state_projection(state_history)
            + self.frame_embedding[None, :, :]
            + self.state_type_embedding
        )
        past_action = (
            self.past_action_projection(past_action_chunks)
            + self.frame_embedding[None, 1:, :]
            + self.past_action_type_embedding
        )

        memory = torch.cat(
            (
                context.flatten(start_dim=1, end_dim=2),
                anchors.flatten(start_dim=1, end_dim=3),
                state,
                past_action,
            ),
            dim=1,
        )
        encoded = self.memory_encoder(memory)
        query = self.action_queries.expand(batch_size, -1, -1) + self.action_position
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            self.action_horizon,
            device=query.device,
            dtype=query.dtype,
        )
        latent = self.action_decoder(query, encoded, tgt_mask=causal_mask)
        delta = torch.tanh(self.delta_head(latent)) * self.step_cap
        current = state_history[:, -1, :]
        bounded_actions = []
        for step in range(self.action_horizon):
            current = torch.maximum(
                torch.minimum(current + delta[:, step], self.action_upper),
                self.action_lower,
            )
            bounded_actions.append(current)
        action = torch.stack(bounded_actions, dim=1)
        action_chunks = action.reshape(
            batch_size,
            self.action_horizon // self.raw_steps_per_chunk,
            self.raw_steps_per_chunk * self.state_dim,
        )
        return SO101GroundedActionOutput(
            action=action,
            delta=delta,
            action_chunks=action_chunks,
            latent=latent,
        )

    def compute_loss(
        self,
        per_view_context: torch.Tensor,
        per_view_anchor_tokens: torch.Tensor,
        anchor_valid: torch.Tensor,
        state_history: torch.Tensor,
        past_action_chunks: torch.Tensor,
        target_action: torch.Tensor,
        target_mask: torch.Tensor,
        *,
        velocity_weight: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], SO101GroundedActionOutput]:
        """Compute masked trajectory and velocity losses."""

        output = self(
            per_view_context,
            per_view_anchor_tokens,
            anchor_valid,
            state_history,
            past_action_chunks,
        )
        if target_action.shape != output.action.shape:
            raise ValueError("target_action must match predicted action shape")
        if target_mask.shape != output.action.shape[:2]:
            raise ValueError("target_mask must match the action time dimensions")

        mask = target_mask.to(dtype=output.action.dtype)
        per_step = torch.mean(torch.square(output.action - target_action), dim=-1)
        action_loss = torch.sum(per_step * mask) / mask.sum().clamp_min(1.0)

        predicted_velocity = torch.diff(
            torch.cat((state_history[:, -1:, :], output.action), dim=1), dim=1
        )
        target_velocity = torch.diff(
            torch.cat((state_history[:, -1:, :], target_action), dim=1), dim=1
        )
        velocity_per_step = torch.mean(
            torch.square(predicted_velocity - target_velocity), dim=-1
        )
        velocity_loss = (
            torch.sum(velocity_per_step * mask) / mask.sum().clamp_min(1.0)
        )
        loss = action_loss + float(velocity_weight) * velocity_loss
        return (
            loss,
            {"action_loss": action_loss, "velocity_loss": velocity_loss},
            output,
        )


# The original pilot used this longer public name. Keep it while the historical
# checkpoint and its cached training script are being reproduced.
SO101GroundedActionChunkPolicy = SO101GroundedActionPolicy


def rollout_frozen_lewm(*args: Any, **kwargs: Any) -> Any:
    """Compatibility placeholder for loading the historical trainer bytecode."""

    raise NotImplementedError("frozen LeWM rollout is not needed for action inference")


def load_so101_grounded_action_policy(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    norm_first: bool = True,
) -> tuple[SO101GroundedActionPolicy, Mapping[str, Any]]:
    """Load the compact real-task action policy and its normalization contract."""

    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "so101-grounded-action-policy-v1":
        raise ValueError(f"unsupported SO-101 action checkpoint: {path}")
    config = dict(checkpoint["model_config"])
    model = SO101GroundedActionPolicy(**config, norm_first=norm_first)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.set_action_contract(
        checkpoint["step_cap"],
        checkpoint["action_lower"],
        checkpoint["action_upper"],
    )
    model.to(device).eval()
    return model, checkpoint


def denormalize_so101_actions(
    values: torch.Tensor | np.ndarray,
    checkpoint: Mapping[str, Any],
) -> torch.Tensor | np.ndarray:
    """Convert checkpoint-normalized actions back to LeRobot joint units."""

    center = np.asarray(checkpoint["normalization_center"], dtype=np.float32)
    scale = np.asarray(checkpoint["normalization_scale"], dtype=np.float32)
    if isinstance(values, torch.Tensor):
        center_t = torch.as_tensor(center, device=values.device, dtype=values.dtype)
        scale_t = torch.as_tensor(scale, device=values.device, dtype=values.dtype)
        return values * scale_t + center_t
    return np.asarray(values) * scale + center


__all__ = [
    "SO101GroundedActionOutput",
    "SO101GroundedActionChunkPolicy",
    "SO101GroundedActionPolicy",
    "denormalize_so101_actions",
    "load_so101_grounded_action_policy",
    "rollout_frozen_lewm",
]
