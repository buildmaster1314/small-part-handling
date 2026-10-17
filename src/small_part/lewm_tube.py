"""Local LeWM-style encoder and latent dynamics, trained without external weights."""
import torch
from torch import nn

class VisualEncoder(nn.Module):
    def __init__(self, width: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, width, 4, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        hidden = self.net(pixels).flatten(1)
        return self.norm(hidden)


class LeWM(nn.Module):
    def __init__(self, action_dim: int = 6, width: int = 128) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.width = width
        self.encoder = VisualEncoder(width)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.predictor = nn.Sequential(
            nn.Linear(width * 2, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        flat = pixels.reshape(-1, *pixels.shape[-3:])
        encoded = self.encoder(flat)
        return encoded.reshape(*pixels.shape[:-3], -1)

    def predict(self, embedding: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        features = torch.cat((embedding, self.action_encoder(action)), dim=-1)
        return self.predictor(features)

    def rollout(self, embedding: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Imagine a latent trajectory. actions is (..., horizon, action_dim)."""

        states = []
        hidden = embedding
        horizon = actions.shape[-2]
        for index in range(horizon):
            hidden = self.predict(hidden, actions[..., index, :])
            states.append(hidden)
        return torch.stack(states, dim=-2)

    def loss(
        self,
        pixels: torch.Tensor,
        action: torch.Tensor,
        next_pixels: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.encode(pixels)
        target = self.encode(next_pixels).detach()
        prediction = self.predict(embedding, action)
        prediction_loss = torch.mean((prediction - target) ** 2)
        variance = embedding.var(dim=0, unbiased=False).clamp_min(1e-6)
        gaussian_loss = torch.mean(variance - variance.log() - 1.0)
        return prediction_loss + 0.02 * gaussian_loss

