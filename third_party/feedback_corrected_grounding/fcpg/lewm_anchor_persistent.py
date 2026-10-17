"""Anchor-persistent LeWM dynamics and candidate innovation scoring.

The legacy grounded LeWM compresses every frame to one summary vector.  This
variant keeps the kinematic-anchor axis through temporal prediction.  The
upstream LeWM predictor and SIGReg objective remain unchanged; only the state
granularity changes from ``[B, T, D]`` to ``[B, T, K, D]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from fcpg.lewm_grounded_model import GroundedFrameEncoder
from third_party.lewm.module import ARPredictor, Embedder, MLP, SIGReg


ANCHOR_PERSISTENT_CHECKPOINT_TYPE = "fcpg_anchor_persistent_lewm_v1"
ANCHOR_PERSISTENT_VARIANT = "anchor_persistent"


def _sequence_tensor(
    batch: Mapping[str, torch.Tensor],
    primary: str,
    fallback: str | None = None,
) -> torch.Tensor:
    value = batch.get(primary)
    if value is None and fallback is not None:
        value = batch.get(fallback)
    if not torch.is_tensor(value):
        names = primary if fallback is None else f"{primary}/{fallback}"
        raise KeyError(f"batch requires tensor {names}")
    return value


@dataclass(frozen=True)
class AnchorPersistentLeWMOutput:
    """Training output compatible with the shared grounded-LeWM runtime."""

    frame_latent: torch.Tensor
    embedding: torch.Tensor
    state_target_embedding: torch.Tensor
    predicted_embedding: torch.Tensor
    target_embedding: torch.Tensor
    prediction_loss: torch.Tensor
    copy_loss: torch.Tensor
    alignment_loss: torch.Tensor
    sigreg_loss: torch.Tensor
    loss: torch.Tensor


class AnchorPersistentLeWM(nn.Module):
    """Action-conditioned LeWM dynamics with a persistent anchor axis.

    Each frame is still encoded jointly, so the per-anchor tokens can attend to
    global visual, robot, and other-anchor evidence.  Temporal prediction then
    treats each anchor as a persistent slot with shared LeWM weights.  This
    preserves anchor identity and time order without multiplying model size by
    the number of anchors.
    """

    checkpoint_type = ANCHOR_PERSISTENT_CHECKPOINT_TYPE
    model_variant = ANCHOR_PERSISTENT_VARIANT

    def __init__(
        self,
        frame_encoder: GroundedFrameEncoder,
        action_dim: int,
        action_smoothing_dim: int = 10,
        history_size: int = 3,
        predictor_depth: int = 6,
        predictor_heads: int = 16,
        predictor_mlp_dim: int = 2048,
        predictor_dim_head: int = 64,
        predictor_dropout: float = 0.1,
        projector_hidden_dim: int = 2048,
        sigreg_knots: int = 17,
        sigreg_num_proj: int = 1024,
        target_mode: str = "online",
        alignment_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        if action_smoothing_dim <= 0:
            raise ValueError("action_smoothing_dim must be positive")
        if target_mode != "online":
            raise ValueError("AnchorPersistentLeWM v1 supports online targets only")
        if float(alignment_weight) != 0.0:
            raise ValueError("AnchorPersistentLeWM v1 requires alignment_weight=0")

        self.frame_encoder = frame_encoder
        self.latent_dim = int(frame_encoder.latent_dim)
        self.num_keypoints = int(frame_encoder.num_keypoints)
        self.action_dim = int(action_dim)
        self.history_size = int(history_size)
        self.target_mode = "online"
        self.alignment_weight = 0.0

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
        self.projector = MLP(
            input_dim=self.latent_dim,
            hidden_dim=int(projector_hidden_dim),
            output_dim=self.latent_dim,
            norm_fn=nn.BatchNorm1d,
        )
        self.pred_proj = MLP(
            input_dim=self.latent_dim,
            hidden_dim=int(projector_hidden_dim),
            output_dim=self.latent_dim,
            norm_fn=nn.BatchNorm1d,
        )
        self.sigreg = SIGReg(knots=int(sigreg_knots), num_proj=int(sigreg_num_proj))
        self.register_buffer(
            "proprio_mean", torch.zeros(self.frame_encoder.obs_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "proprio_std", torch.ones(self.frame_encoder.obs_dim, dtype=torch.float32)
        )
        self.register_buffer("action_mean", torch.zeros(self.action_dim, dtype=torch.float32))
        self.register_buffer("action_std", torch.ones(self.action_dim, dtype=torch.float32))

    def set_input_normalization(
        self,
        *,
        proprio_mean: torch.Tensor | list[float],
        proprio_std: torch.Tensor | list[float],
        action_mean: torch.Tensor | list[float],
        action_std: torch.Tensor | list[float],
    ) -> None:
        values = {
            "proprio_mean": (proprio_mean, self.frame_encoder.obs_dim),
            "proprio_std": (proprio_std, self.frame_encoder.obs_dim),
            "action_mean": (action_mean, self.action_dim),
            "action_std": (action_std, self.action_dim),
        }
        with torch.no_grad():
            for name, (value, size) in values.items():
                target = getattr(self, name)
                tensor = torch.as_tensor(value, dtype=target.dtype, device=target.device)
                if tensor.shape != (size,):
                    raise ValueError(
                        f"{name} must have shape {(size,)}, got {tuple(tensor.shape)}"
                    )
                if name.endswith("_std") and bool(torch.any(tensor <= 0.0)):
                    raise ValueError(f"{name} must be strictly positive")
                target.copy_(tensor)

    @staticmethod
    def _flatten_sequence(
        value: torch.Tensor,
        batch_size: int,
        sequence_length: int,
        trailing_shape: tuple[int, ...],
        name: str,
    ) -> torch.Tensor:
        expected = (batch_size, sequence_length, *trailing_shape)
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")
        return value.reshape(batch_size * sequence_length, *trailing_shape)

    def project_anchor_tokens(self, anchor_tokens: torch.Tensor) -> torch.Tensor:
        """Project frame-encoder anchor tokens while retaining leading axes."""

        if anchor_tokens.ndim < 3:
            raise ValueError("anchor_tokens must have shape [..., K, D]")
        if anchor_tokens.shape[-2:] != (self.num_keypoints, self.latent_dim):
            raise ValueError(
                "anchor_tokens must end with "
                f"{(self.num_keypoints, self.latent_dim)}, got {tuple(anchor_tokens.shape)}"
            )
        projected = self.projector(anchor_tokens.reshape(-1, self.latent_dim))
        return projected.reshape(*anchor_tokens.shape)

    def encode_frames(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = _sequence_tensor(batch, "pixels", "rgb")
        if pixels.ndim != 5 or pixels.shape[2] != 3:
            raise ValueError(
                f"pixels/rgb must have shape [B, T, 3, H, W], got {tuple(pixels.shape)}"
            )
        batch_size, sequence_length, _, height, width = pixels.shape
        obs = _sequence_tensor(batch, "proprio", "obs")
        q_nominal = _sequence_tensor(batch, "q_nominal")
        q_grounded = batch.get("q_corrected")
        if q_grounded is None:
            q_grounded = batch.get("q_grounded")
        if q_grounded is None:
            q_grounded = q_nominal
        if not torch.is_tensor(q_grounded):
            raise TypeError("q_grounded must be a tensor")
        keypoints_xyz = _sequence_tensor(batch, "keypoints_xyz")
        valid = batch.get("valid")
        if valid is None:
            valid = batch.get("valid_nominal")

        flat_pixels = self._flatten_sequence(
            pixels, batch_size, sequence_length, (3, height, width), "pixels"
        ).float()
        if flat_pixels.numel() and flat_pixels.detach().amax() > 1.5:
            flat_pixels = flat_pixels / 255.0
        flat_obs = self._flatten_sequence(
            obs,
            batch_size,
            sequence_length,
            (self.frame_encoder.obs_dim,),
            "proprio/obs",
        ).float()
        flat_obs = (flat_obs - self.proprio_mean) / self.proprio_std
        flat_q_nominal = self._flatten_sequence(
            q_nominal,
            batch_size,
            sequence_length,
            (self.num_keypoints, 2),
            "q_nominal",
        ).float()
        flat_q_grounded = self._flatten_sequence(
            q_grounded,
            batch_size,
            sequence_length,
            (self.num_keypoints, 2),
            "q_corrected/q_grounded",
        ).float()
        flat_xyz = self._flatten_sequence(
            keypoints_xyz,
            batch_size,
            sequence_length,
            (self.num_keypoints, 3),
            "keypoints_xyz",
        ).float()
        flat_valid = None
        if valid is not None:
            if not torch.is_tensor(valid):
                raise TypeError("valid must be a tensor")
            flat_valid = self._flatten_sequence(
                valid,
                batch_size,
                sequence_length,
                (self.num_keypoints,),
                "valid",
            ).bool()

        tokens = self.frame_encoder.forward_token_sequence(
            pixels=flat_pixels,
            robot_obs=flat_obs,
            q_nominal_px=flat_q_nominal,
            q_grounded_px=flat_q_grounded,
            keypoints_xyz=flat_xyz,
            valid=flat_valid,
        )
        anchor_tokens = tokens[:, -self.num_keypoints :]
        anchor_tokens = anchor_tokens.reshape(
            batch_size, sequence_length, self.num_keypoints, self.latent_dim
        )
        return anchor_tokens, self.project_anchor_tokens(anchor_tokens)

    def predict(self, embedding: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if embedding.ndim != 4 or embedding.shape[-2:] != (
            self.num_keypoints,
            self.latent_dim,
        ):
            raise ValueError(
                "embedding must have shape "
                f"[B, T, {self.num_keypoints}, {self.latent_dim}], "
                f"got {tuple(embedding.shape)}"
            )
        batch_size, sequence_length = embedding.shape[:2]
        if action.shape != (batch_size, sequence_length, self.action_dim):
            raise ValueError(
                "action must have shape "
                f"{(batch_size, sequence_length, self.action_dim)}, "
                f"got {tuple(action.shape)}"
            )

        flat_embedding = embedding.permute(0, 2, 1, 3).reshape(
            batch_size * self.num_keypoints, sequence_length, self.latent_dim
        )
        normalized_action = (action - self.action_mean) / self.action_std
        flat_action = normalized_action[:, None].expand(
            -1, self.num_keypoints, -1, -1
        ).reshape(batch_size * self.num_keypoints, sequence_length, self.action_dim)
        predicted = self.predictor(flat_embedding, self.action_encoder(flat_action))
        predicted = self.pred_proj(predicted.reshape(-1, self.latent_dim))
        return predicted.reshape(
            batch_size, self.num_keypoints, sequence_length, self.latent_dim
        ).permute(0, 2, 1, 3)

    def _sigreg_loss(
        self,
        embedding: torch.Tensor,
        source_index: torch.Tensor | None,
    ) -> torch.Tensor:
        if source_index is not None:
            if source_index.ndim == 2:
                if not bool(torch.all(source_index == source_index[:, :1])):
                    raise ValueError("source_index must be constant within each sequence")
                source_index = source_index[:, 0]
            if source_index.ndim != 1 or source_index.shape[0] != embedding.shape[0]:
                raise ValueError("source_index must have shape [B] or [B, T]")

        losses = []
        for anchor in range(self.num_keypoints):
            anchor_embedding = embedding[:, :, anchor]
            if source_index is None:
                losses.append(self.sigreg(anchor_embedding.transpose(0, 1)))
                continue
            for source in torch.unique(source_index):
                selected = anchor_embedding[source_index == source]
                if selected.shape[0] >= 2:
                    losses.append(self.sigreg(selected.transpose(0, 1)))
        if not losses:
            return embedding.sum() * 0.0
        return torch.stack(losses).mean()

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        sigreg_weight: float = 0.02,
        per_source_sigreg: bool = False,
    ) -> AnchorPersistentLeWMOutput:
        frame_latent, embedding = self.encode_frames(batch)
        required = self.history_size + 1
        if embedding.shape[1] != required:
            raise ValueError(
                f"faithful LeWM training expects exactly H+1={required} frames, "
                f"got {embedding.shape[1]}"
            )
        action = _sequence_tensor(batch, "action", "actions").float()
        if action.ndim != 3 or action.shape[:2] != (
            embedding.shape[0],
            self.history_size,
        ) or action.shape[-1] != self.action_dim:
            raise ValueError(
                "action must have shape "
                f"[B, {self.history_size}, {self.action_dim}], got {tuple(action.shape)}"
            )
        context = embedding[:, : self.history_size]
        target = embedding[:, 1 : self.history_size + 1]
        predicted = self.predict(context, action)
        prediction_loss = F.mse_loss(predicted, target)
        copy_loss = F.mse_loss(embedding[:, : self.history_size], target)
        alignment_loss = embedding.sum() * 0.0
        source_index = batch.get("source_index") if per_source_sigreg else None
        sigreg_loss = self._sigreg_loss(embedding, source_index)
        loss = prediction_loss + float(sigreg_weight) * sigreg_loss
        return AnchorPersistentLeWMOutput(
            frame_latent=frame_latent,
            embedding=embedding,
            state_target_embedding=embedding,
            predicted_embedding=predicted,
            target_embedding=target,
            prediction_loss=prediction_loss,
            copy_loss=copy_loss,
            alignment_loss=alignment_loss,
            sigreg_loss=sigreg_loss,
            loss=loss,
        )

    def rollout_anchor_embeddings(
        self,
        history_embedding: torch.Tensor,
        history_action: torch.Tensor,
        future_action: torch.Tensor,
    ) -> torch.Tensor:
        """Recursively predict ``[B, S, P, K, D]`` anchor embeddings."""

        if history_embedding.ndim != 4 or history_embedding.shape[1:] != (
            self.history_size,
            self.num_keypoints,
            self.latent_dim,
        ):
            raise ValueError(
                "history_embedding must have shape "
                f"[B, {self.history_size}, {self.num_keypoints}, {self.latent_dim}]"
            )
        batch_size = history_embedding.shape[0]
        if history_action.shape != (
            batch_size,
            self.history_size - 1,
            self.action_dim,
        ):
            raise ValueError("history_action has an invalid shape")
        if future_action.ndim != 4 or future_action.shape[0] != batch_size:
            raise ValueError("future_action must have shape [B, S, P, A]")
        if future_action.shape[-1] != self.action_dim:
            raise ValueError("future_action has an invalid action dimension")

        samples, predictions = future_action.shape[1:3]
        embeddings = history_embedding[:, None].expand(-1, samples, -1, -1, -1)
        actions = history_action[:, None].expand(-1, samples, -1, -1)
        outputs = []
        for step in range(predictions):
            action_context = torch.cat(
                [actions, future_action[:, :, step : step + 1]], dim=2
            )
            flat_embedding = embeddings.reshape(
                batch_size * samples,
                self.history_size,
                self.num_keypoints,
                self.latent_dim,
            )
            flat_action = action_context.reshape(
                batch_size * samples, self.history_size, self.action_dim
            )
            next_embedding = self.predict(flat_embedding, flat_action)[:, -1]
            next_embedding = next_embedding.reshape(
                batch_size, samples, 1, self.num_keypoints, self.latent_dim
            )
            outputs.append(next_embedding)
            embeddings = torch.cat([embeddings[:, :, 1:], next_embedding], dim=2)
            actions = action_context[:, :, 1:]
        if not outputs:
            return history_embedding.new_empty(
                batch_size, samples, 0, self.num_keypoints, self.latent_dim
            )
        return torch.cat(outputs, dim=2)


def anchor_innovation_features(
    candidate_embedding: torch.Tensor,
    predicted_embedding: torch.Tensor,
) -> torch.Tensor:
    """Construct explicit candidate-vs-prior features per persistent anchor.

    Args:
        candidate_embedding: ``[B, M, K, D]`` candidate measurements.
        predicted_embedding: ``[B, K, D]`` action-conditioned prior.
    """

    if candidate_embedding.ndim != 4:
        raise ValueError("candidate_embedding must have shape [B, M, K, D]")
    if predicted_embedding.shape != (
        candidate_embedding.shape[0],
        candidate_embedding.shape[2],
        candidate_embedding.shape[3],
    ):
        raise ValueError("predicted_embedding must match candidate [B, K, D]")
    prior = predicted_embedding[:, None].expand_as(candidate_embedding)
    delta = candidate_embedding - prior
    squared_distance = delta.square().mean(dim=-1, keepdim=True)
    cosine = F.cosine_similarity(candidate_embedding, prior, dim=-1).unsqueeze(-1)
    return torch.cat(
        [candidate_embedding, prior, delta, delta.abs(), candidate_embedding * prior,
         squared_distance, cosine],
        dim=-1,
    )


class AnchorInnovationScorer(nn.Module):
    """Small candidate scorer that cannot erase anchor identity by early pooling."""

    def __init__(
        self,
        latent_dim: int,
        num_keypoints: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or num_keypoints <= 0 or hidden_dim <= 0:
            raise ValueError("scorer dimensions must be positive")
        self.latent_dim = int(latent_dim)
        self.num_keypoints = int(num_keypoints)
        feature_dim = 5 * self.latent_dim + 2
        self.anchor_identity = nn.Parameter(
            torch.zeros(1, 1, self.num_keypoints, hidden_dim)
        )
        self.anchor_encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.trunc_normal_(self.anchor_identity, std=0.02)

    def forward(
        self,
        candidate_embedding: torch.Tensor,
        predicted_embedding: torch.Tensor,
        candidate_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = anchor_innovation_features(candidate_embedding, predicted_embedding)
        if features.shape[2] != self.num_keypoints or features.shape[3] != (
            5 * self.latent_dim + 2
        ):
            raise ValueError("candidate embeddings do not match scorer dimensions")
        encoded = self.anchor_encoder(features) + self.anchor_identity
        if candidate_valid is None:
            pooled = encoded.mean(dim=2)
        else:
            if candidate_valid.shape != candidate_embedding.shape[:3]:
                raise ValueError("candidate_valid must have shape [B, M, K]")
            weights = candidate_valid.to(dtype=encoded.dtype).unsqueeze(-1)
            pooled = (encoded * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        return self.score_head(pooled).squeeze(-1)
