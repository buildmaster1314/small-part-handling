"""Pixel-space innovation observer coupling FCPG measurements to LeWM priors.

The legacy grounded LeWM replaces its latent history with every new encoded
frame, which is equivalent to trusting the current visual measurement with unit
gain.  This module makes the missing observer state explicit:

``d_measurement = q_corrected - q_nominal``
``innovation = d_measurement - d_prior``
``d_posterior = d_prior + alpha * innovation``

The action-conditioned LeWM predicts the next latent.  A small, zero-initialized
adapter decodes that latent and the last kinematic anchor state into a pixel
residual prior.  After the measurement update, the frame is sampled again at
``q_nominal + d_posterior`` and the resulting embedding is written back into
the temporal history.  No logged future observation is accepted by the prior
API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from fcpg.lewm_grounded_model import GroundedLeWM, _sequence_tensor
from fcpg.modules import sample_feature_map_at_pixels


PIXEL_OBSERVER_CHECKPOINT_TYPE = "fcpg_lewm_pixel_observer_v1"


def _gain_tensor(
    gain: float | torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    value = torch.as_tensor(gain, dtype=reference.dtype, device=reference.device)
    if value.numel() == 0 or not bool(torch.isfinite(value).all()):
        raise ValueError("observer gain must contain only finite values")
    if bool(torch.any(value < 0.0)) or bool(torch.any(value > 1.0)):
        raise ValueError("observer gain must lie in the closed interval [0, 1]")
    try:
        return torch.broadcast_to(value, reference.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"observer gain with shape {tuple(value.shape)} is not broadcastable "
            f"to residual shape {tuple(reference.shape)}"
        ) from exc


def fixed_gain_update(
    prior_residual_px: torch.Tensor,
    measurement_residual_px: torch.Tensor,
    gain: float | torch.Tensor,
    measurement_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one fixed-gain innovation update in pixel-residual space.

    Args:
        prior_residual_px: Prior residual with final coordinate dimension two.
        measurement_residual_px: FCPG residual with the exact same shape.
        gain: Scalar or broadcastable tensor in ``[0, 1]``.
        measurement_valid: Optional mask matching the residual shape without its
            final coordinate dimension.  Invalid measurements receive zero gain.

    Returns:
        ``(innovation, posterior, effective_gain)``.  The effective gain has the
        full residual shape, which makes masked endpoint semantics auditable.
    """

    if prior_residual_px.shape != measurement_residual_px.shape:
        raise ValueError(
            "prior and measurement residuals must have identical shapes: "
            f"{tuple(prior_residual_px.shape)} != "
            f"{tuple(measurement_residual_px.shape)}"
        )
    if prior_residual_px.ndim < 2 or prior_residual_px.shape[-1] != 2:
        raise ValueError(
            "pixel residuals must end in a two-dimensional coordinate axis"
        )
    if (
        not prior_residual_px.is_floating_point()
        or not measurement_residual_px.is_floating_point()
    ):
        raise TypeError("pixel residuals must use floating point tensors")
    if not bool(torch.isfinite(prior_residual_px).all()):
        raise ValueError("prior residuals must contain only finite values")
    effective_gain = _gain_tensor(gain, prior_residual_px)
    if measurement_valid is not None:
        expected = prior_residual_px.shape[:-1]
        if measurement_valid.shape != expected:
            raise ValueError(
                f"measurement_valid must have shape {expected}, got "
                f"{tuple(measurement_valid.shape)}"
            )
        valid_coordinates = measurement_valid.bool().unsqueeze(-1).expand_as(
            measurement_residual_px
        )
        if bool(
            (valid_coordinates & ~torch.isfinite(measurement_residual_px)).any()
        ):
            raise ValueError("valid measurement residuals must be finite")
        effective_gain = torch.where(
            valid_coordinates, effective_gain, torch.zeros_like(effective_gain)
        )
        innovation = torch.where(
            valid_coordinates,
            measurement_residual_px - prior_residual_px,
            torch.zeros_like(prior_residual_px),
        )
    else:
        if not bool(torch.isfinite(measurement_residual_px).all()):
            raise ValueError("measurement residuals must contain only finite values")
        innovation = measurement_residual_px - prior_residual_px
    posterior = prior_residual_px + effective_gain * innovation
    return innovation, posterior, effective_gain


@dataclass(frozen=True)
class PixelObserverStep:
    """One prediction-correction step of the explicit pixel observer."""

    prior_embedding: torch.Tensor
    measurement_residual_px: torch.Tensor
    prior_residual_px: torch.Tensor
    innovation_px: torch.Tensor
    posterior_residual_px: torch.Tensor
    q_posterior_px: torch.Tensor
    effective_gain: torch.Tensor


@dataclass(frozen=True)
class PixelObserverSequence:
    """Observed filtering result after a shared warm-up history."""

    warmup_frames: int
    measurement_residual_px: torch.Tensor
    prior_residual_px: torch.Tensor
    innovation_px: torch.Tensor
    posterior_residual_px: torch.Tensor
    q_posterior_px: torch.Tensor
    effective_gain: torch.Tensor
    posterior_embedding: torch.Tensor
    prior_loss: torch.Tensor
    hold_last_loss: torch.Tensor


@dataclass(frozen=True)
class PixelObserverForecast:
    """Open-loop residual forecast with no logged future observation input."""

    prior_residual_px: torch.Tensor
    predicted_embedding: torch.Tensor
    geometry_rollout: str = "last_observed_zero_order_hold"


class PixelResidualPriorAdapter(nn.Module):
    """Decode a LeWM latent prior into an anchor-wise 2D residual prior.

    The final layer is zero initialized and predicts a bounded increment over
    the last posterior residual.  An untrained adapter is therefore exactly the
    hold-last residual baseline rather than an arbitrary zero projection.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        num_keypoints: int,
        image_size: tuple[int, int],
        hidden_dim: int = 256,
        max_delta_px: float = 16.0,
        residual_scale_px: float = 16.0,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or num_keypoints <= 0 or hidden_dim <= 0:
            raise ValueError("adapter dimensions must be positive")
        width, height = (int(image_size[0]), int(image_size[1]))
        if width <= 1 or height <= 1:
            raise ValueError("image_size must contain dimensions larger than one")
        if max_delta_px <= 0.0 or residual_scale_px <= 0.0:
            raise ValueError("pixel scales must be positive")
        self.latent_dim = int(latent_dim)
        self.num_keypoints = int(num_keypoints)
        self.image_size = (width, height)
        self.max_delta_px = float(max_delta_px)
        self.residual_scale_px = float(residual_scale_px)

        self.anchor_embedding = nn.Parameter(
            torch.zeros(1, self.num_keypoints, self.latent_dim)
        )
        # latent + residual(2) + nominal pixel(2) + metric XYZ(3) + valid(1)
        input_dim = self.latent_dim + 8
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.anchor_embedding, std=0.02)

        self.register_buffer("xyz_mean", torch.zeros(3, dtype=torch.float32))
        self.register_buffer("xyz_std", torch.ones(3, dtype=torch.float32))

    def set_geometry_normalization(
        self,
        *,
        xyz_mean: torch.Tensor | list[float],
        xyz_std: torch.Tensor | list[float],
    ) -> None:
        mean = torch.as_tensor(
            xyz_mean, dtype=self.xyz_mean.dtype, device=self.xyz_mean.device
        )
        std = torch.as_tensor(
            xyz_std, dtype=self.xyz_std.dtype, device=self.xyz_std.device
        )
        if mean.shape != (3,) or std.shape != (3,):
            raise ValueError("xyz_mean and xyz_std must each have shape [3]")
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
            raise ValueError("geometry normalization must be finite")
        if bool(torch.any(std <= 0.0)):
            raise ValueError("xyz_std must be strictly positive")
        with torch.no_grad():
            self.xyz_mean.copy_(mean)
            self.xyz_std.copy_(std)

    def _normalized_nominal(self, q_nominal_px: torch.Tensor) -> torch.Tensor:
        width, height = self.image_size
        scale = q_nominal_px.new_tensor((float(width - 1), float(height - 1)))
        return 2.0 * q_nominal_px / scale - 1.0

    def forward(
        self,
        predicted_embedding: torch.Tensor,
        last_posterior_residual_px: torch.Tensor,
        last_q_nominal_px: torch.Tensor,
        last_keypoints_xyz: torch.Tensor,
        last_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = predicted_embedding.shape[0]
        expected_residual = (batch_size, self.num_keypoints, 2)
        expected_xyz = (batch_size, self.num_keypoints, 3)
        expected_valid = (batch_size, self.num_keypoints)
        if predicted_embedding.shape != (batch_size, self.latent_dim):
            raise ValueError(
                "predicted_embedding must have shape "
                f"{(batch_size, self.latent_dim)}, got "
                f"{tuple(predicted_embedding.shape)}"
            )
        for name, value in (
            ("last_posterior_residual_px", last_posterior_residual_px),
            ("last_q_nominal_px", last_q_nominal_px),
        ):
            if value.shape != expected_residual:
                raise ValueError(
                    f"{name} must have shape {expected_residual}, got "
                    f"{tuple(value.shape)}"
                )
        if last_keypoints_xyz.shape != expected_xyz:
            raise ValueError(
                f"last_keypoints_xyz must have shape {expected_xyz}, got "
                f"{tuple(last_keypoints_xyz.shape)}"
            )
        if last_valid.shape != expected_valid:
            raise ValueError(
                f"last_valid must have shape {expected_valid}, got "
                f"{tuple(last_valid.shape)}"
            )

        latent = predicted_embedding.unsqueeze(1) + self.anchor_embedding
        xyz = (last_keypoints_xyz - self.xyz_mean) / self.xyz_std
        features = torch.cat(
            [
                latent,
                last_posterior_residual_px / self.residual_scale_px,
                self._normalized_nominal(last_q_nominal_px),
                xyz,
                last_valid.to(dtype=last_q_nominal_px.dtype).unsqueeze(-1),
            ],
            dim=-1,
        )
        delta = self.max_delta_px * torch.tanh(self.net(features))
        return last_posterior_residual_px + delta


class LeWMPixelInnovationObserver(nn.Module):
    """Closed-loop pixel observer around a frozen grounded LeWM.

    Only the residual adapter is trainable by default.  The base LeWM supplies
    an action-conditioned latent prior; the FCPG correction cache supplies the
    visual measurement.  Target-frame RGB and kinematics are consumed only
    after the prior has been computed.
    """

    def __init__(
        self,
        base_model: GroundedLeWM,
        adapter: PixelResidualPriorAdapter,
        *,
        gain: float = 0.5,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if adapter.latent_dim != base_model.latent_dim:
            raise ValueError("adapter latent_dim must match the base LeWM")
        if adapter.num_keypoints != base_model.frame_encoder.num_keypoints:
            raise ValueError("adapter keypoint count must match the base LeWM")
        self.base_model = base_model
        self.adapter = adapter
        self.default_gain = float(gain)
        _gain_tensor(self.default_gain, torch.zeros(1, 1, 2))
        self.freeze_base = bool(freeze_base)
        if self.freeze_base:
            self.base_model.requires_grad_(False)
            self.base_model.eval()

    @property
    def history_size(self) -> int:
        return int(self.base_model.history_size)

    @property
    def num_keypoints(self) -> int:
        return int(self.adapter.num_keypoints)

    def train(self, mode: bool = True) -> "LeWMPixelInnovationObserver":
        super().train(mode)
        if self.freeze_base:
            self.base_model.eval()
        return self

    def encode_observed_frames(
        self,
        batch: Mapping[str, torch.Tensor],
        q_posterior_px: torch.Tensor,
    ) -> torch.Tensor:
        """Encode real frames sampled at posterior anchors."""

        encoded_batch = dict(batch)
        encoded_batch["q_corrected"] = q_posterior_px
        encoded_batch["valid"] = self._posterior_valid(batch, q_posterior_px)
        _, embedding = self.base_model.encode_frames(encoded_batch)
        return embedding

    def _posterior_valid(
        self,
        batch: Mapping[str, torch.Tensor],
        q_posterior_px: torch.Tensor,
    ) -> torch.Tensor:
        base_valid = batch.get("valid_nominal")
        if base_valid is None:
            base_valid = batch.get("valid")
        if not torch.is_tensor(base_valid) or base_valid.shape != q_posterior_px.shape[:-1]:
            raise ValueError("valid_nominal/valid must match posterior anchor shape")
        width, height = self.adapter.image_size
        inside = (
            (q_posterior_px[..., 0] >= 0.0)
            & (q_posterior_px[..., 0] <= float(width - 1))
            & (q_posterior_px[..., 1] >= 0.0)
            & (q_posterior_px[..., 1] <= float(height - 1))
        )
        return base_valid.bool() & inside

    def predict_prior(
        self,
        *,
        history_embedding: torch.Tensor,
        transition_actions: torch.Tensor,
        last_posterior_residual_px: torch.Tensor,
        last_q_nominal_px: torch.Tensor,
        last_keypoints_xyz: torch.Tensor,
        last_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict one prior without accepting any future observation tensor."""

        batch_size = history_embedding.shape[0]
        if history_embedding.shape != (
            batch_size,
            self.history_size,
            self.base_model.latent_dim,
        ):
            raise ValueError(
                "history_embedding must have shape "
                f"[B, {self.history_size}, {self.base_model.latent_dim}]"
            )
        if transition_actions.shape != (
            batch_size,
            self.history_size,
            self.base_model.action_dim,
        ):
            raise ValueError(
                "transition_actions must contain H-1 history actions plus the "
                "current transition action"
            )
        predicted_embedding = self.base_model.predict(
            history_embedding,
            transition_actions,
        )[:, -1]
        prior_residual = self.adapter(
            predicted_embedding,
            last_posterior_residual_px,
            last_q_nominal_px,
            last_keypoints_xyz,
            last_valid,
        )
        return predicted_embedding, prior_residual

    def update(
        self,
        *,
        prior_embedding: torch.Tensor,
        prior_residual_px: torch.Tensor,
        q_nominal_px: torch.Tensor,
        q_corrected_px: torch.Tensor,
        gain: float | torch.Tensor | None = None,
        measurement_valid: torch.Tensor | None = None,
    ) -> PixelObserverStep:
        measurement = q_corrected_px - q_nominal_px
        innovation, posterior, effective_gain = fixed_gain_update(
            prior_residual_px,
            measurement,
            self.default_gain if gain is None else gain,
            measurement_valid,
        )
        return PixelObserverStep(
            prior_embedding=prior_embedding,
            measurement_residual_px=measurement,
            prior_residual_px=prior_residual_px,
            innovation_px=innovation,
            posterior_residual_px=posterior,
            q_posterior_px=q_nominal_px + posterior,
            effective_gain=effective_gain,
        )

    def _validate_filter_batch(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[int, int]:
        forbidden = sorted(
            name for name in ("q_true", "q_oracle", "oracle_grounding") if name in batch
        )
        if forbidden:
            raise ValueError(
                "oracle pixel coordinates are evaluator-only and must not enter "
                f"the observer: {forbidden}"
            )
        q_nominal = _sequence_tensor(batch, "q_nominal")
        q_corrected = _sequence_tensor(batch, "q_corrected", "q_grounded")
        if q_nominal.shape != q_corrected.shape or q_nominal.ndim != 4:
            raise ValueError("q_nominal and q_corrected must have shape [B, T, K, 2]")
        batch_size, frames, keypoints, coordinates = q_nominal.shape
        if keypoints != self.num_keypoints or coordinates != 2:
            raise ValueError("observer grounding dimensions do not match the adapter")
        if frames <= self.history_size:
            raise ValueError("filtering requires at least H+1 observed frames")
        actions = _sequence_tensor(batch, "action", "actions")
        if actions.shape != (batch_size, frames - 1, self.base_model.action_dim):
            raise ValueError("action must contain one transition for each frame pair")
        return batch_size, frames

    @staticmethod
    def _slice_frames(
        batch: Mapping[str, torch.Tensor],
        start: int,
        stop: int,
    ) -> dict[str, torch.Tensor]:
        frame_names = {
            "pixels",
            "rgb",
            "obs",
            "proprio",
            "q_nominal",
            "q_corrected",
            "q_grounded",
            "keypoints_xyz",
            "valid",
            "valid_nominal",
        }
        return {
            name: value[:, start:stop]
            for name, value in batch.items()
            if name in frame_names and torch.is_tensor(value)
        }

    def filter_sequence(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        gain: float | torch.Tensor | None = None,
    ) -> PixelObserverSequence:
        """Filter a real observed sequence and write every posterior into history.

        The first ``H`` frames form a common FCPG-measurement warm-up for all
        gain values.  Metrics and losses start at frame ``H``.  At each later
        frame, the prior is computed before any tensor from that frame is used.
        """

        _, frames = self._validate_filter_batch(batch)
        q_nominal = _sequence_tensor(batch, "q_nominal").float()
        q_corrected = _sequence_tensor(batch, "q_corrected", "q_grounded").float()
        keypoints_xyz = _sequence_tensor(batch, "keypoints_xyz").float()
        measurement_valid = batch.get("valid")
        if measurement_valid is None:
            measurement_valid = batch.get("valid_nominal")
        if (
            not torch.is_tensor(measurement_valid)
            or measurement_valid.shape != q_nominal.shape[:-1]
        ):
            raise ValueError("valid must have shape [B, T, K]")
        measurement_valid = measurement_valid.bool()
        actions = _sequence_tensor(batch, "action", "actions").float()
        measurement = q_corrected - q_nominal

        warmup_posteriors: list[torch.Tensor] = []
        warmup_posterior = torch.zeros_like(measurement[:, 0])
        for warmup_index in range(self.history_size):
            _, warmup_posterior, _ = fixed_gain_update(
                warmup_posterior,
                measurement[:, warmup_index],
                1.0,
                measurement_valid[:, warmup_index],
            )
            warmup_posteriors.append(warmup_posterior)
        history_posteriors = torch.stack(warmup_posteriors, dim=1)
        warmup_batch = self._slice_frames(batch, 0, self.history_size)
        history_embedding = self.encode_observed_frames(
            warmup_batch,
            q_nominal[:, : self.history_size] + history_posteriors,
        )
        history_valid = self._posterior_valid(
            warmup_batch,
            q_nominal[:, : self.history_size] + history_posteriors,
        )

        priors: list[torch.Tensor] = []
        innovations: list[torch.Tensor] = []
        posteriors: list[torch.Tensor] = []
        q_posteriors: list[torch.Tensor] = []
        gains: list[torch.Tensor] = []
        posterior_embeddings: list[torch.Tensor] = []

        for target_index in range(self.history_size, frames):
            action_start = target_index - self.history_size
            action_context = actions[:, action_start:target_index]
            predicted_embedding, prior = self.predict_prior(
                history_embedding=history_embedding,
                transition_actions=action_context,
                last_posterior_residual_px=history_posteriors[:, -1],
                last_q_nominal_px=q_nominal[:, target_index - 1],
                last_keypoints_xyz=keypoints_xyz[:, target_index - 1],
                last_valid=history_valid[:, -1],
            )
            step = self.update(
                prior_embedding=predicted_embedding,
                prior_residual_px=prior,
                q_nominal_px=q_nominal[:, target_index],
                q_corrected_px=q_corrected[:, target_index],
                gain=gain,
                measurement_valid=measurement_valid[:, target_index],
            )

            target_batch = self._slice_frames(batch, target_index, target_index + 1)
            target_posterior_valid = self._posterior_valid(
                target_batch,
                step.q_posterior_px.unsqueeze(1),
            )
            target_embedding = self.encode_observed_frames(
                target_batch,
                step.q_posterior_px.unsqueeze(1),
            )[:, 0]
            history_embedding = torch.cat(
                [history_embedding[:, 1:], target_embedding.unsqueeze(1)], dim=1
            )
            history_posteriors = torch.cat(
                [
                    history_posteriors[:, 1:],
                    step.posterior_residual_px.unsqueeze(1),
                ],
                dim=1,
            )
            history_valid = torch.cat(
                [history_valid[:, 1:], target_posterior_valid], dim=1
            )

            priors.append(step.prior_residual_px)
            innovations.append(step.innovation_px)
            posteriors.append(step.posterior_residual_px)
            q_posteriors.append(step.q_posterior_px)
            gains.append(step.effective_gain)
            posterior_embeddings.append(target_embedding)

        prior_sequence = torch.stack(priors, dim=1)
        innovation_sequence = torch.stack(innovations, dim=1)
        posterior_sequence = torch.stack(posteriors, dim=1)
        q_posterior_sequence = torch.stack(q_posteriors, dim=1)
        gain_sequence = torch.stack(gains, dim=1)
        embedding_sequence = torch.stack(posterior_embeddings, dim=1)
        target_measurement = measurement[:, self.history_size :]
        target_valid = measurement_valid[:, self.history_size :].unsqueeze(-1)
        prior_loss = _masked_mse(prior_sequence, target_measurement, target_valid)
        hold_last_steps: list[torch.Tensor] = []
        last_valid_measurement = history_posteriors[:, -1]
        for target_index in range(self.history_size, frames):
            hold_last_steps.append(last_valid_measurement)
            _, last_valid_measurement, _ = fixed_gain_update(
                last_valid_measurement,
                measurement[:, target_index],
                1.0,
                measurement_valid[:, target_index],
            )
        hold_last = torch.stack(hold_last_steps, dim=1)
        hold_last_loss = _masked_mse(hold_last, target_measurement, target_valid)
        return PixelObserverSequence(
            warmup_frames=self.history_size,
            measurement_residual_px=target_measurement,
            prior_residual_px=prior_sequence,
            innovation_px=innovation_sequence,
            posterior_residual_px=posterior_sequence,
            q_posterior_px=q_posterior_sequence,
            effective_gain=gain_sequence,
            posterior_embedding=embedding_sequence,
            prior_loss=prior_loss,
            hold_last_loss=hold_last_loss,
        )

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        gain: float | torch.Tensor | None = None,
    ) -> PixelObserverSequence:
        return self.filter_sequence(batch, gain=gain)

    def forecast_residuals(
        self,
        history_batch: Mapping[str, torch.Tensor],
        history_posterior_residual_px: torch.Tensor,
        future_action: torch.Tensor,
    ) -> PixelObserverForecast:
        """Recursively forecast residuals without logged future observations.

        Metric geometry is held at its last observed value.  This explicit
        zero-order hold is intentionally conservative: accepting logged future
        FK, RGB, nominal projections, or FCPG measurements here would leak the
        evaluation trajectory into a planning-time forecast.
        """

        forbidden = sorted(
            name
            for name in ("q_true", "q_oracle", "oracle_grounding")
            if name in history_batch
        )
        if forbidden:
            raise ValueError(
                "oracle pixel coordinates are evaluator-only and must not enter "
                f"the observer: {forbidden}"
            )
        q_nominal = _sequence_tensor(history_batch, "q_nominal").float()
        keypoints_xyz = _sequence_tensor(history_batch, "keypoints_xyz").float()
        batch_size = q_nominal.shape[0]
        expected_residual = (
            batch_size,
            self.history_size,
            self.num_keypoints,
            2,
        )
        if q_nominal.shape != expected_residual:
            raise ValueError(
                f"history q_nominal must have shape {expected_residual}, got "
                f"{tuple(q_nominal.shape)}"
            )
        if history_posterior_residual_px.shape != expected_residual:
            raise ValueError(
                "history_posterior_residual_px must match history q_nominal"
            )
        if keypoints_xyz.shape != (
            batch_size,
            self.history_size,
            self.num_keypoints,
            3,
        ):
            raise ValueError("history keypoints_xyz shape is invalid")
        history_action = _sequence_tensor(history_batch, "action", "actions").float()
        if history_action.shape != (
            batch_size,
            self.history_size - 1,
            self.base_model.action_dim,
        ):
            raise ValueError("history action must contain exactly H-1 transitions")
        if (
            future_action.ndim != 3
            or future_action.shape[0] != batch_size
            or future_action.shape[2] != self.base_model.action_dim
        ):
            raise ValueError("future_action must have shape [B, P, action_dim]")

        q_posterior = q_nominal + history_posterior_residual_px
        history_embedding = self.encode_observed_frames(history_batch, q_posterior)
        history_valid = self._posterior_valid(history_batch, q_posterior)
        last_residual = history_posterior_residual_px[:, -1]
        last_q_nominal = q_nominal[:, -1]
        last_xyz = keypoints_xyz[:, -1]
        last_valid = history_valid[:, -1]
        residual_outputs: list[torch.Tensor] = []
        embedding_outputs: list[torch.Tensor] = []

        for step in range(future_action.shape[1]):
            current_action = future_action[:, step : step + 1]
            action_context = torch.cat([history_action, current_action], dim=1)
            predicted_embedding, prior = self.predict_prior(
                history_embedding=history_embedding,
                transition_actions=action_context,
                last_posterior_residual_px=last_residual,
                last_q_nominal_px=last_q_nominal,
                last_keypoints_xyz=last_xyz,
                last_valid=last_valid,
            )
            residual_outputs.append(prior)
            embedding_outputs.append(predicted_embedding)
            history_embedding = torch.cat(
                [history_embedding[:, 1:], predicted_embedding.unsqueeze(1)], dim=1
            )
            history_action = action_context[:, 1:]
            last_residual = prior

        if not residual_outputs:
            empty_residual = q_nominal.new_empty(
                batch_size, 0, self.num_keypoints, 2
            )
            empty_embedding = history_embedding.new_empty(
                batch_size, 0, self.base_model.latent_dim
            )
            return PixelObserverForecast(empty_residual, empty_embedding)
        return PixelObserverForecast(
            prior_residual_px=torch.stack(residual_outputs, dim=1),
            predicted_embedding=torch.stack(embedding_outputs, dim=1),
        )


def _masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("masked MSE prediction and target shapes must match")
    if mask.shape != prediction.shape[:-1] + (1,):
        raise ValueError("masked MSE mask shape is incompatible with prediction")
    selected = mask.bool().expand_as(prediction)
    difference = torch.where(
        selected, prediction - target, torch.zeros_like(prediction)
    )
    if not bool(torch.isfinite(difference).all()):
        raise ValueError("masked MSE contains a non-finite selected value")
    weights = selected.to(dtype=prediction.dtype)
    denominator = weights.sum().clamp_min(1.0)
    return difference.square().sum() / denominator


def sample_posterior_features(
    feature_map: torch.Tensor,
    q_nominal_px: torch.Tensor,
    posterior_residual_px: torch.Tensor,
    *,
    image_size: tuple[int, int],
    padding_mode: str = "border",
) -> torch.Tensor:
    """Sample a dense feature map at explicit posterior anchor locations."""

    if q_nominal_px.shape != posterior_residual_px.shape:
        raise ValueError("q_nominal_px and posterior_residual_px shapes must match")
    return sample_feature_map_at_pixels(
        feature_map,
        q_nominal_px + posterior_residual_px,
        image_size=image_size,
        padding_mode=padding_mode,
    )
