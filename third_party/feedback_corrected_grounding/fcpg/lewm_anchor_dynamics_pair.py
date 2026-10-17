"""Matched action-conditioned and action-independent LeWM predictor heads.

The heads operate in one frozen Anchor-Persistent LeWM latent space.  They
share architecture and initialization; the only experimental factor is
whether the action sequence is visible to the predictor during training.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from third_party.lewm.module import ARPredictor, Embedder, MLP


MATCHED_DYNAMICS_PAIR_CHECKPOINT_TYPE = "fcpg_matched_anchor_dynamics_pair_v1"


@dataclass(frozen=True)
class MatchedDynamicsOutput:
    predicted: torch.Tensor
    target: torch.Tensor
    prediction_loss: torch.Tensor


class MatchedAnchorDynamicsPredictor(nn.Module):
    """LeWM dynamics head over frozen persistent-anchor embeddings."""

    def __init__(
        self,
        *,
        latent_dim: int,
        num_keypoints: int,
        action_dim: int,
        history_size: int,
        condition_on_action: bool,
        action_smoothing_dim: int = 10,
        predictor_depth: int = 6,
        predictor_heads: int = 16,
        predictor_mlp_dim: int = 2048,
        predictor_dim_head: int = 64,
        predictor_dropout: float = 0.1,
        projector_hidden_dim: int = 2048,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or num_keypoints <= 0 or action_dim <= 0:
            raise ValueError("latent, keypoint, and action dimensions must be positive")
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        self.latent_dim = int(latent_dim)
        self.num_keypoints = int(num_keypoints)
        self.action_dim = int(action_dim)
        self.history_size = int(history_size)
        self.condition_on_action = bool(condition_on_action)

        self.predictor = ARPredictor(
            num_frames=self.history_size,
            depth=int(predictor_depth),
            heads=int(predictor_heads),
            mlp_dim=int(predictor_mlp_dim),
            input_dim=self.latent_dim,
            hidden_dim=self.latent_dim,
            output_dim=self.latent_dim,
            dim_head=int(predictor_dim_head),
            dropout=float(predictor_dropout),
            emb_dropout=0.0,
        )
        self.action_encoder = Embedder(
            input_dim=self.action_dim,
            smoothed_dim=int(action_smoothing_dim),
            emb_dim=self.latent_dim,
        )
        self.pred_proj = MLP(
            input_dim=self.latent_dim,
            hidden_dim=int(projector_hidden_dim),
            output_dim=self.latent_dim,
            norm_fn=nn.BatchNorm1d,
        )
        self.register_buffer("action_mean", torch.zeros(self.action_dim))
        self.register_buffer("action_std", torch.ones(self.action_dim))

    def set_action_normalization(
        self,
        *,
        action_mean: torch.Tensor | list[float],
        action_std: torch.Tensor | list[float],
    ) -> None:
        mean = torch.as_tensor(
            action_mean, dtype=self.action_mean.dtype, device=self.action_mean.device
        )
        std = torch.as_tensor(
            action_std, dtype=self.action_std.dtype, device=self.action_std.device
        )
        expected = (self.action_dim,)
        if mean.shape != expected or std.shape != expected:
            raise ValueError(f"action normalization must have shape {expected}")
        if bool(torch.any(std <= 0.0)):
            raise ValueError("action_std must be strictly positive")
        with torch.no_grad():
            self.action_mean.copy_(mean)
            self.action_std.copy_(std)

    def predict(self, embedding: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        expected_embedding_tail = (self.num_keypoints, self.latent_dim)
        if embedding.ndim != 4 or embedding.shape[-2:] != expected_embedding_tail:
            raise ValueError(
                "embedding must have shape "
                f"[B, T, {self.num_keypoints}, {self.latent_dim}]"
            )
        batch_size, sequence_length = embedding.shape[:2]
        expected_action = (batch_size, sequence_length, self.action_dim)
        if action.shape != expected_action:
            raise ValueError(
                f"action must have shape {expected_action}, got {tuple(action.shape)}"
            )

        flat_embedding = embedding.permute(0, 2, 1, 3).reshape(
            batch_size * self.num_keypoints,
            sequence_length,
            self.latent_dim,
        )
        normalized_action = (action.float() - self.action_mean) / self.action_std
        if not self.condition_on_action:
            normalized_action = torch.zeros_like(normalized_action)
        flat_action = normalized_action[:, None].expand(
            -1, self.num_keypoints, -1, -1
        ).reshape(
            batch_size * self.num_keypoints,
            sequence_length,
            self.action_dim,
        )
        predicted = self.predictor(
            flat_embedding,
            self.action_encoder(flat_action),
        )
        predicted = self.pred_proj(predicted.reshape(-1, self.latent_dim))
        return predicted.reshape(
            batch_size,
            self.num_keypoints,
            sequence_length,
            self.latent_dim,
        ).permute(0, 2, 1, 3)

    def forward(
        self,
        context: torch.Tensor,
        action: torch.Tensor,
        target: torch.Tensor,
    ) -> MatchedDynamicsOutput:
        predicted = self.predict(context, action)
        if target.shape != predicted.shape:
            raise ValueError(
                f"target must have shape {tuple(predicted.shape)}, got {tuple(target.shape)}"
            )
        loss = F.mse_loss(predicted, target)
        return MatchedDynamicsOutput(
            predicted=predicted,
            target=target,
            prediction_loss=loss,
        )


def predictor_kwargs_from_base_checkpoint(
    checkpoint: Mapping[str, object],
) -> dict[str, object]:
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("base checkpoint is missing model_config")
    frame = model_config.get("frame")
    world = model_config.get("world_model")
    if not isinstance(frame, Mapping) or not isinstance(world, Mapping):
        raise ValueError("base checkpoint model_config is incomplete")
    return {
        "latent_dim": int(frame["latent_dim"]),
        "num_keypoints": int(frame["num_keypoints"]),
        "action_dim": int(world["action_dim"]),
        "history_size": int(world["history_size"]),
        "action_smoothing_dim": int(world["action_smoothing_dim"]),
        "predictor_depth": int(world["predictor_depth"]),
        "predictor_heads": int(world["predictor_heads"]),
        "predictor_mlp_dim": int(world["predictor_mlp_dim"]),
        "predictor_dim_head": int(world["predictor_dim_head"]),
        "predictor_dropout": float(world["predictor_dropout"]),
        "projector_hidden_dim": int(world["projector_hidden_dim"]),
    }


def build_matched_predictor_pair(
    checkpoint: Mapping[str, object],
    *,
    initialization_seed: int,
) -> tuple[MatchedAnchorDynamicsPredictor, MatchedAnchorDynamicsPredictor]:
    """Build byte-identical heads before exposing actions to only one head."""

    kwargs = predictor_kwargs_from_base_checkpoint(checkpoint)
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(initialization_seed))
        factual = MatchedAnchorDynamicsPredictor(
            **kwargs,
            condition_on_action=True,
        )
    no_action = deepcopy(factual)
    no_action.condition_on_action = False
    return factual, no_action


def matched_pair_forward(
    factual: MatchedAnchorDynamicsPredictor,
    no_action: MatchedAnchorDynamicsPredictor,
    context: torch.Tensor,
    action: torch.Tensor,
    target: torch.Tensor,
) -> tuple[MatchedDynamicsOutput, MatchedDynamicsOutput]:
    """Evaluate both heads with identical stochastic-layer random draws."""

    cpu_state = torch.random.get_rng_state()
    cuda_state = (
        torch.cuda.get_rng_state(context.device)
        if context.device.type == "cuda"
        else None
    )
    factual_output = factual(context, action, target)
    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, context.device)
    no_action_output = no_action(context, action, target)
    return factual_output, no_action_output


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_matched_predictor_pair_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[
    MatchedAnchorDynamicsPredictor,
    MatchedAnchorDynamicsPredictor,
    dict[str, object],
]:
    checkpoint = torch.load(
        Path(path).expanduser().resolve(),
        map_location=torch.device(device),
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("predictor-pair checkpoint must contain a mapping")
    if checkpoint.get("checkpoint_type") != MATCHED_DYNAMICS_PAIR_CHECKPOINT_TYPE:
        raise ValueError("checkpoint is not a matched anchor dynamics pair")
    config = checkpoint.get("predictor_config")
    if not isinstance(config, Mapping):
        raise ValueError("predictor-pair checkpoint is missing predictor_config")
    factual = MatchedAnchorDynamicsPredictor(
        **dict(config), condition_on_action=True
    )
    no_action = MatchedAnchorDynamicsPredictor(
        **dict(config), condition_on_action=False
    )
    factual.load_state_dict(checkpoint["factual_state"], strict=True)
    no_action.load_state_dict(checkpoint["no_action_state"], strict=True)
    target_device = torch.device(device)
    factual.to(target_device).eval()
    no_action.to(target_device).eval()
    expected_count = checkpoint.get("parameter_count_per_head")
    if expected_count is not None:
        actual_counts = (
            trainable_parameter_count(factual),
            trainable_parameter_count(no_action),
        )
        if any(count != int(expected_count) for count in actual_counts):
            raise ValueError("predictor-pair checkpoint parameter count is inconsistent")
    return factual, no_action, checkpoint


__all__ = [
    "MATCHED_DYNAMICS_PAIR_CHECKPOINT_TYPE",
    "MatchedAnchorDynamicsPredictor",
    "MatchedDynamicsOutput",
    "build_matched_predictor_pair",
    "load_matched_predictor_pair_checkpoint",
    "matched_pair_forward",
    "predictor_kwargs_from_base_checkpoint",
    "trainable_parameter_count",
]
