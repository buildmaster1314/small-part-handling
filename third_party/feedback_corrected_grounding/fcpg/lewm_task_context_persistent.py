"""Task-conditioned LeWM dynamics with persistent summary and anchor slots.

This variant retains the grounded frame encoder's summary token alongside its
three kinematic-anchor tokens.  Temporal prediction shares weights across the
four state slots.  Task identity conditions the action path of the upstream
LeWM predictor and is never appended to the predicted or SIGReg state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from fcpg.lewm_grounded_model import GroundedFrameEncoder
from third_party.lewm.module import ARPredictor, Embedder, MLP, SIGReg


TASK_CONTEXT_PERSISTENT_CHECKPOINT_TYPE = "fcpg_task_context_persistent_lewm_v1"
TASK_CONTEXT_PERSISTENT_VARIANT = "task_context_persistent"
TASK_CONTEXT_PERSISTENT_SCHEMA_VERSION = 1
TASK_CONTEXT_PERSISTENT_SIGREG_WEIGHT = 0.02
TASK_CONTEXT_PERSISTENT_CONTEXT_WEIGHT = 0.25
TASK_CONTEXT_PERSISTENT_ANCHOR_WEIGHT = 0.75
TASK_CONTEXT_PERSISTENT_STATE_SLOTS = (
    "visual_summary",
    "anchor0",
    "anchor1",
    "anchor2",
)

_NUM_KEYPOINTS = 3
_NUM_STATE_SLOTS = len(TASK_CONTEXT_PERSISTENT_STATE_SLOTS)
_INTEGER_DTYPES = frozenset(
    {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


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
class TaskContextPersistentLeWMOutput:
    """Training output compatible with the grounded-LeWM output contract."""

    frame_latent: torch.Tensor
    embedding: torch.Tensor
    state_target_embedding: torch.Tensor
    predicted_embedding: torch.Tensor
    target_embedding: torch.Tensor
    prediction_loss: torch.Tensor
    context_loss: torch.Tensor
    anchor_loss: torch.Tensor
    copy_loss: torch.Tensor
    alignment_loss: torch.Tensor
    sigreg_loss: torch.Tensor
    loss: torch.Tensor

    @property
    def context_prediction_loss(self) -> torch.Tensor:
        return self.context_loss

    @property
    def anchor_prediction_loss(self) -> torch.Tensor:
        return self.anchor_loss


class TaskContextPersistentLeWM(nn.Module):
    """Shared LeWM dynamics over one summary and three persistent anchors."""

    checkpoint_type = TASK_CONTEXT_PERSISTENT_CHECKPOINT_TYPE
    model_variant = TASK_CONTEXT_PERSISTENT_VARIANT
    state_slot_names = TASK_CONTEXT_PERSISTENT_STATE_SLOTS
    sigreg_weight = TASK_CONTEXT_PERSISTENT_SIGREG_WEIGHT
    context_loss_weight = TASK_CONTEXT_PERSISTENT_CONTEXT_WEIGHT
    anchor_loss_weight = TASK_CONTEXT_PERSISTENT_ANCHOR_WEIGHT

    def __init__(
        self,
        frame_encoder: GroundedFrameEncoder,
        action_dim: int,
        num_tasks: int,
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
        if int(frame_encoder.num_keypoints) != _NUM_KEYPOINTS:
            raise ValueError(
                "TaskContextPersistentLeWM requires num_keypoints=3, got "
                f"{frame_encoder.num_keypoints}"
            )
        action_dim = _positive_int(action_dim, "action_dim")
        num_tasks = _positive_int(num_tasks, "num_tasks")
        history_size = _positive_int(history_size, "history_size")
        action_smoothing_dim = _positive_int(
            action_smoothing_dim,
            "action_smoothing_dim",
        )
        if target_mode != "online":
            raise ValueError(
                "TaskContextPersistentLeWM v1 supports online targets only"
            )
        if float(alignment_weight) != 0.0:
            raise ValueError(
                "TaskContextPersistentLeWM v1 requires alignment_weight=0"
            )

        self.frame_encoder = frame_encoder
        self.latent_dim = int(frame_encoder.latent_dim)
        self.num_keypoints = _NUM_KEYPOINTS
        self.num_state_slots = _NUM_STATE_SLOTS
        self.action_dim = action_dim
        self.num_tasks = num_tasks
        self.history_size = history_size
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
            smoothed_dim=action_smoothing_dim,
            emb_dim=self.latent_dim,
        )
        self.task_embedding = nn.Embedding(self.num_tasks, self.latent_dim)
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
            "proprio_mean",
            torch.zeros(self.frame_encoder.obs_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "proprio_std",
            torch.ones(self.frame_encoder.obs_dim, dtype=torch.float32),
        )
        self.register_buffer("action_mean", torch.zeros(self.action_dim))
        self.register_buffer("action_std", torch.ones(self.action_dim))
        self.register_buffer(
            "state_schema_version",
            torch.tensor(TASK_CONTEXT_PERSISTENT_SCHEMA_VERSION, dtype=torch.int64),
            persistent=True,
        )

    def set_input_normalization(
        self,
        *,
        proprio_mean: torch.Tensor | list[float],
        proprio_std: torch.Tensor | list[float],
        action_mean: torch.Tensor | list[float],
        action_std: torch.Tensor | list[float],
    ) -> None:
        """Install train-split normalization statistics."""

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

    def select_state_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Select ``[summary, anchor0, anchor1, anchor2]`` encoder tokens."""

        expected_tokens = 3 + self.num_keypoints
        if tokens.ndim != 3 or tokens.shape[1:] != (
            expected_tokens,
            self.latent_dim,
        ):
            raise ValueError(
                "tokens must have shape "
                f"[B, {expected_tokens}, {self.latent_dim}], got {tuple(tokens.shape)}"
            )
        return torch.cat(
            [tokens[:, 0:1], tokens[:, -self.num_keypoints :]],
            dim=1,
        )

    def project_state_tokens(self, state_tokens: torch.Tensor) -> torch.Tensor:
        """Project four state slots while retaining all leading dimensions."""

        if state_tokens.ndim < 3:
            raise ValueError("state_tokens must have shape [..., 4, D]")
        if state_tokens.shape[-2:] != (self.num_state_slots, self.latent_dim):
            raise ValueError(
                "state_tokens must end with "
                f"{(self.num_state_slots, self.latent_dim)}, "
                f"got {tuple(state_tokens.shape)}"
            )
        projected = self.projector(state_tokens.reshape(-1, self.latent_dim))
        return projected.reshape(*state_tokens.shape)

    def encode_frames(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode observed frames into unprojected and projected four-slot state."""

        pixels = _sequence_tensor(batch, "pixels", "rgb")
        if pixels.ndim != 5 or pixels.shape[2] != 3:
            raise ValueError(
                f"pixels/rgb must have shape [B, T, 3, H, W], got {tuple(pixels.shape)}"
            )
        batch_size, sequence_length, _, height, width = pixels.shape
        if batch_size <= 0 or sequence_length <= 0:
            raise ValueError("pixels/rgb batch and sequence dimensions must be positive")
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
            pixels,
            batch_size,
            sequence_length,
            (3, height, width),
            "pixels",
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
        state_tokens = self.select_state_tokens(tokens).reshape(
            batch_size,
            sequence_length,
            self.num_state_slots,
            self.latent_dim,
        )
        return state_tokens, self.project_state_tokens(state_tokens)

    def _validated_task_index(
        self,
        task_index: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not torch.is_tensor(task_index):
            raise TypeError("task_index must be a tensor")
        if task_index.shape != (batch_size,):
            raise ValueError(
                f"task_index must have shape {(batch_size,)}, "
                f"got {tuple(task_index.shape)}"
            )
        if task_index.dtype not in _INTEGER_DTYPES:
            raise TypeError("task_index must have an integer dtype")
        if task_index.numel() == 0:
            raise ValueError("task_index must not be empty")
        minimum = int(task_index.detach().amin().item())
        maximum = int(task_index.detach().amax().item())
        if minimum < 0 or maximum >= self.num_tasks:
            raise ValueError(
                f"task_index values must be in [0, {self.num_tasks - 1}], "
                f"got range [{minimum}, {maximum}]"
            )
        return task_index.to(device=device, dtype=torch.long)

    def predict(
        self,
        embedding: torch.Tensor,
        action: torch.Tensor,
        task_index: torch.Tensor,
    ) -> torch.Tensor:
        """Predict each state slot using action and task-conditioned dynamics."""

        if embedding.ndim != 4 or embedding.shape[-2:] != (
            self.num_state_slots,
            self.latent_dim,
        ):
            raise ValueError(
                "embedding must have shape "
                f"[B, T, {self.num_state_slots}, {self.latent_dim}], "
                f"got {tuple(embedding.shape)}"
            )
        batch_size, sequence_length = embedding.shape[:2]
        if batch_size <= 0 or not 0 < sequence_length <= self.history_size:
            raise ValueError(
                f"embedding sequence length must be in [1, {self.history_size}]"
            )
        if action.shape != (batch_size, sequence_length, self.action_dim):
            raise ValueError(
                "action must have shape "
                f"{(batch_size, sequence_length, self.action_dim)}, "
                f"got {tuple(action.shape)}"
            )
        task_index = self._validated_task_index(
            task_index,
            batch_size,
            embedding.device,
        )

        flat_embedding = embedding.permute(0, 2, 1, 3).reshape(
            batch_size * self.num_state_slots,
            sequence_length,
            self.latent_dim,
        )
        normalized_action = (action - self.action_mean) / self.action_std
        flat_action = normalized_action[:, None].expand(
            -1,
            self.num_state_slots,
            -1,
            -1,
        ).reshape(
            batch_size * self.num_state_slots,
            sequence_length,
            self.action_dim,
        )
        task_condition = self.task_embedding(task_index)
        flat_task_condition = task_condition[:, None].expand(
            -1,
            self.num_state_slots,
            -1,
        ).reshape(batch_size * self.num_state_slots, self.latent_dim)
        predictor_condition = self.action_encoder(flat_action)
        predictor_condition = predictor_condition + flat_task_condition[:, None]
        predicted = self.predictor(flat_embedding, predictor_condition)
        predicted = self.pred_proj(predicted.reshape(-1, self.latent_dim))
        return predicted.reshape(
            batch_size,
            self.num_state_slots,
            sequence_length,
            self.latent_dim,
        ).permute(0, 2, 1, 3)

    @staticmethod
    def _source_vector(
        source_index: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.is_tensor(source_index):
            raise TypeError("source_index must be a tensor")
        if source_index.ndim == 2:
            if source_index.shape != embedding.shape[:2]:
                raise ValueError("source_index must have shape [B] or [B, T]")
            if not bool(torch.all(source_index == source_index[:, :1])):
                raise ValueError("source_index must be constant within each sequence")
            source_index = source_index[:, 0]
        if source_index.ndim != 1 or source_index.shape[0] != embedding.shape[0]:
            raise ValueError("source_index must have shape [B] or [B, T]")
        return source_index.to(device=embedding.device)

    def _sigreg_loss(
        self,
        embedding: torch.Tensor,
        source_index: torch.Tensor | None,
    ) -> torch.Tensor:
        if embedding.ndim != 4 or embedding.shape[-2:] != (
            self.num_state_slots,
            self.latent_dim,
        ):
            raise ValueError(
                "embedding must have shape "
                f"[B, T, {self.num_state_slots}, {self.latent_dim}]"
            )
        sources = (
            None
            if source_index is None
            else self._source_vector(source_index, embedding)
        )
        slot_losses = []
        for slot in range(self.num_state_slots):
            slot_embedding = embedding[:, :, slot]
            if sources is None:
                slot_losses.append(self.sigreg(slot_embedding.transpose(0, 1)))
                continue
            source_losses = []
            for source in torch.unique(sources):
                selected = slot_embedding[sources == source]
                if selected.shape[0] >= 2:
                    source_losses.append(self.sigreg(selected.transpose(0, 1)))
            if source_losses:
                slot_losses.append(torch.stack(source_losses).mean())
            else:
                slot_losses.append(slot_embedding.sum() * 0.0)
        return torch.stack(slot_losses).mean()

    @staticmethod
    def _validate_sigreg_weight(sigreg_weight: float) -> float:
        try:
            value = float(sigreg_weight)
        except (TypeError, ValueError) as exc:
            raise TypeError("sigreg_weight must be numeric") from exc
        if not math.isfinite(value) or value != TASK_CONTEXT_PERSISTENT_SIGREG_WEIGHT:
            raise ValueError(
                "TaskContextPersistentLeWM v1 requires sigreg_weight exactly 0.02"
            )
        return value

    @staticmethod
    def _weighted_slot_mse(
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context_loss = F.mse_loss(predicted[:, :, 0], target[:, :, 0])
        anchor_loss = F.mse_loss(predicted[:, :, 1:], target[:, :, 1:])
        combined = (
            TASK_CONTEXT_PERSISTENT_CONTEXT_WEIGHT * context_loss
            + TASK_CONTEXT_PERSISTENT_ANCHOR_WEIGHT * anchor_loss
        )
        return combined, context_loss, anchor_loss

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        sigreg_weight: float = TASK_CONTEXT_PERSISTENT_SIGREG_WEIGHT,
        per_source_sigreg: bool = False,
    ) -> TaskContextPersistentLeWMOutput:
        """Compute the exact ``H + 1`` teacher-forced objective."""

        sigreg_weight = self._validate_sigreg_weight(sigreg_weight)
        frame_latent, embedding = self.encode_frames(batch)
        required = self.history_size + 1
        if embedding.shape[1] != required:
            raise ValueError(
                f"faithful LeWM training expects exactly H+1={required} frames, "
                f"got {embedding.shape[1]}"
            )
        action = _sequence_tensor(batch, "action", "actions").float()
        if action.shape != (
            embedding.shape[0],
            self.history_size,
            self.action_dim,
        ):
            raise ValueError(
                "action must have shape "
                f"{(embedding.shape[0], self.history_size, self.action_dim)}, "
                f"got {tuple(action.shape)}"
            )
        task_index = _sequence_tensor(batch, "task_index")
        context = embedding[:, : self.history_size]
        target = embedding[:, 1 : self.history_size + 1]
        predicted = self.predict(context, action, task_index)
        prediction_loss, context_loss, anchor_loss = self._weighted_slot_mse(
            predicted,
            target,
        )
        copy_loss, _, _ = self._weighted_slot_mse(context, target)
        alignment_loss = embedding.sum() * 0.0
        source_index = batch.get("source_index") if per_source_sigreg else None
        sigreg_loss = self._sigreg_loss(embedding, source_index)
        loss = prediction_loss + sigreg_weight * sigreg_loss
        return TaskContextPersistentLeWMOutput(
            frame_latent=frame_latent,
            embedding=embedding,
            state_target_embedding=embedding,
            predicted_embedding=predicted,
            target_embedding=target,
            prediction_loss=prediction_loss,
            context_loss=context_loss,
            anchor_loss=anchor_loss,
            copy_loss=copy_loss,
            alignment_loss=alignment_loss,
            sigreg_loss=sigreg_loss,
            loss=loss,
        )

    def rollout_state_embeddings(
        self,
        history_embedding: torch.Tensor,
        history_action: torch.Tensor,
        future_action: torch.Tensor,
        task_index: torch.Tensor,
    ) -> torch.Tensor:
        """Recursively predict ``[B, S, P, 4, D]`` state embeddings."""

        if history_embedding.ndim != 4 or history_embedding.shape[1:] != (
            self.history_size,
            self.num_state_slots,
            self.latent_dim,
        ):
            raise ValueError(
                "history_embedding must have shape "
                f"[B, {self.history_size}, {self.num_state_slots}, "
                f"{self.latent_dim}], got {tuple(history_embedding.shape)}"
            )
        batch_size = history_embedding.shape[0]
        if batch_size <= 0:
            raise ValueError("history_embedding batch dimension must be positive")
        if history_action.shape != (
            batch_size,
            self.history_size - 1,
            self.action_dim,
        ):
            raise ValueError(
                "history_action must have shape "
                f"{(batch_size, self.history_size - 1, self.action_dim)}, "
                f"got {tuple(history_action.shape)}"
            )
        if future_action.ndim != 4 or future_action.shape[0] != batch_size:
            raise ValueError("future_action must have shape [B, S, P, A]")
        if future_action.shape[-1] != self.action_dim:
            raise ValueError(
                f"future_action last dimension must be {self.action_dim}, "
                f"got {future_action.shape[-1]}"
            )
        task_index = self._validated_task_index(
            task_index,
            batch_size,
            history_embedding.device,
        )

        samples, predictions = future_action.shape[1:3]
        if samples == 0 or predictions == 0:
            return history_embedding.new_empty(
                batch_size,
                samples,
                predictions,
                self.num_state_slots,
                self.latent_dim,
            )
        embeddings = history_embedding[:, None].expand(-1, samples, -1, -1, -1)
        actions = history_action[:, None].expand(-1, samples, -1, -1)
        flat_task_index = task_index[:, None].expand(-1, samples).reshape(-1)
        outputs = []
        for step in range(predictions):
            action_context = torch.cat(
                [actions, future_action[:, :, step : step + 1]],
                dim=2,
            )
            flat_embedding = embeddings.reshape(
                batch_size * samples,
                self.history_size,
                self.num_state_slots,
                self.latent_dim,
            )
            flat_action = action_context.reshape(
                batch_size * samples,
                self.history_size,
                self.action_dim,
            )
            next_embedding = self.predict(
                flat_embedding,
                flat_action,
                flat_task_index,
            )[:, -1]
            next_embedding = next_embedding.reshape(
                batch_size,
                samples,
                1,
                self.num_state_slots,
                self.latent_dim,
            )
            outputs.append(next_embedding)
            embeddings = torch.cat([embeddings[:, :, 1:], next_embedding], dim=2)
            actions = action_context[:, :, 1:]
        return torch.cat(outputs, dim=2)

    def rollout_embeddings(
        self,
        history_embedding: torch.Tensor,
        history_action: torch.Tensor,
        future_action: torch.Tensor,
        task_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compatibility alias for :meth:`rollout_state_embeddings`."""

        return self.rollout_state_embeddings(
            history_embedding,
            history_action,
            future_action,
            task_index,
        )


__all__ = [
    "TASK_CONTEXT_PERSISTENT_ANCHOR_WEIGHT",
    "TASK_CONTEXT_PERSISTENT_CHECKPOINT_TYPE",
    "TASK_CONTEXT_PERSISTENT_CONTEXT_WEIGHT",
    "TASK_CONTEXT_PERSISTENT_SCHEMA_VERSION",
    "TASK_CONTEXT_PERSISTENT_SIGREG_WEIGHT",
    "TASK_CONTEXT_PERSISTENT_STATE_SLOTS",
    "TASK_CONTEXT_PERSISTENT_VARIANT",
    "TaskContextPersistentLeWM",
    "TaskContextPersistentLeWMOutput",
]
