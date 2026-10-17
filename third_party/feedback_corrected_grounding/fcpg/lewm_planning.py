"""Planning utilities for the geometry-grounded LeWM latent dynamics model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class LatentDynamicsModel(Protocol):
    def rollout_embeddings(
        self,
        history_embedding: torch.Tensor,
        history_action: torch.Tensor,
        future_action: torch.Tensor,
    ) -> torch.Tensor: ...

    def goal_cost(
        self,
        predicted_embedding: torch.Tensor,
        goal_embedding: torch.Tensor,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class CEMPlan:
    action_sequence: torch.Tensor
    cost: torch.Tensor
    distribution_mean: torch.Tensor
    distribution_std: torch.Tensor


class GroundedCEMPlanner:
    """Cross-entropy planner over recursive LeWM latent rollouts.

    The planner is environment-independent.  A MetaWorld rollout wrapper only
    needs to maintain the last ``history_size`` grounded observations, encode a
    visual goal, and execute the first ``receding_horizon`` actions from the
    returned sequence.
    """

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        num_samples: int = 512,
        num_elites: int = 64,
        iterations: int = 5,
        momentum: float = 0.1,
        min_std: float = 0.05,
        action_low: float | tuple[float, ...] = -1.0,
        action_high: float | tuple[float, ...] = 1.0,
    ) -> None:
        if horizon <= 0 or action_dim <= 0:
            raise ValueError("horizon and action_dim must be positive")
        if not 0 < num_elites <= num_samples:
            raise ValueError("num_elites must be in [1, num_samples]")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.num_samples = int(num_samples)
        self.num_elites = int(num_elites)
        self.iterations = int(iterations)
        self.momentum = float(momentum)
        self.min_std = float(min_std)
        self.action_low = action_low
        self.action_high = action_high

    def _bound(
        self,
        value: float | tuple[float, ...],
        reference: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        result = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        if result.ndim == 0:
            result = result.expand(self.action_dim)
        if result.shape != (self.action_dim,):
            raise ValueError(f"{name} must be scalar or have shape {(self.action_dim,)}")
        return result

    @torch.no_grad()
    def plan(
        self,
        model: LatentDynamicsModel,
        history_embedding: torch.Tensor,
        history_action: torch.Tensor,
        goal_embedding: torch.Tensor,
        initial_mean: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> CEMPlan:
        if history_embedding.ndim != 3:
            raise ValueError("history_embedding must have shape [B, H, D]")
        batch_size = history_embedding.shape[0]
        low = self._bound(self.action_low, history_embedding, "action_low")
        high = self._bound(self.action_high, history_embedding, "action_high")
        if not bool(torch.all(low < high)):
            raise ValueError("every action_low value must be below action_high")
        shape = (batch_size, self.horizon, self.action_dim)
        if initial_mean is None:
            mean = ((low + high) / 2.0).view(1, 1, -1).expand(shape).clone()
        else:
            if initial_mean.shape != shape:
                raise ValueError(f"initial_mean must have shape {shape}")
            mean = initial_mean.clone().clamp(low, high)
        std = ((high - low) / 2.0).view(1, 1, -1).expand(shape).clone()
        low_view = low.view(1, 1, 1, -1)
        high_view = high.view(1, 1, 1, -1)

        for _ in range(self.iterations):
            noise = torch.randn(
                batch_size,
                self.num_samples,
                self.horizon,
                self.action_dim,
                device=history_embedding.device,
                dtype=history_embedding.dtype,
                generator=generator,
            )
            candidates = mean[:, None] + std[:, None] * noise
            candidates = candidates.clamp(low_view, high_view)
            predicted = model.rollout_embeddings(
                history_embedding,
                history_action,
                candidates,
            )
            cost = model.goal_cost(predicted, goal_embedding)
            elite_index = torch.topk(
                cost,
                k=self.num_elites,
                dim=1,
                largest=False,
            ).indices
            gather_index = elite_index[..., None, None].expand(
                -1,
                -1,
                self.horizon,
                self.action_dim,
            )
            elite = torch.gather(candidates, dim=1, index=gather_index)
            elite_mean = elite.mean(dim=1)
            elite_std = elite.std(dim=1, unbiased=False).clamp_min(self.min_std)
            mean = self.momentum * mean + (1.0 - self.momentum) * elite_mean
            std = self.momentum * std + (1.0 - self.momentum) * elite_std

        action_sequence = mean.clamp(low, high)
        predicted = model.rollout_embeddings(
            history_embedding,
            history_action,
            action_sequence[:, None],
        )
        final_cost = model.goal_cost(predicted, goal_embedding)[:, 0]
        return CEMPlan(
            action_sequence=action_sequence,
            cost=final_cost,
            distribution_mean=mean,
            distribution_std=std,
        )

