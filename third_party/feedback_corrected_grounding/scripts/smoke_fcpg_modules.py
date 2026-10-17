#!/usr/bin/env python3
"""Smoke checks for reusable FCPG PyTorch modules."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.modules import (  # noqa: E402
    FCPGResidualOnly,
    GroundedTokenAdapter,
    ReliabilityConfidenceHead,
    ResidualCorrectionHead,
    sample_feature_map_at_pixels,
)


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def run_feature_sampling_check(device: torch.device) -> None:
    feature = torch.zeros(1, 2, 8, 8, device=device)
    yy, xx = torch.meshgrid(torch.arange(8, device=device), torch.arange(8, device=device), indexing="ij")
    feature[0, 0] = xx.float()
    feature[0, 1] = yy.float()
    points = torch.tensor([[[0.0, 0.0], [3.0, 4.0], [7.0, 6.0]]], device=device)
    sampled = sample_feature_map_at_pixels(feature, points, image_size=(8, 8))
    expected = torch.tensor([[[0.0, 0.0], [3.0, 4.0], [7.0, 6.0]]], device=device)
    assert torch.allclose(sampled, expected, atol=1e-5), (sampled, expected)

    feature_grad = feature.clone().detach().requires_grad_(True)
    sampled_grad = sample_feature_map_at_pixels(feature_grad, points, image_size=(8, 8))
    sampled_grad.sum().backward()
    assert feature_grad.grad is not None
    assert float(feature_grad.grad.abs().sum().item()) > 0.0
    print("feature_sampling_check: PASS")


def run_grounded_token_check(device: torch.device) -> None:
    batch, num_points, visual_dim, proprio_dim, token_dim = 2, 3, 4, 5, 7
    feature_map = torch.randn(batch, visual_dim, 16, 16, device=device, requires_grad=True)
    points = torch.tensor(
        [
            [[2.0, 3.0], [7.0, 8.0], [12.0, 5.0]],
            [[1.0, 1.0], [4.0, 10.0], [14.0, 13.0]],
        ],
        device=device,
    )
    proprio = torch.randn(batch, num_points, proprio_dim, device=device)
    reliability = torch.tensor([[1.0, 0.5, 0.0], [0.25, 0.75, 1.0]], device=device)
    adapter = GroundedTokenAdapter(visual_dim, proprio_dim, token_dim, gate_visual_only=True).to(device)
    tokens = adapter(feature_map, points, proprio, image_size=(16, 16), reliability=reliability)
    assert tuple(tokens.shape) == (batch, num_points, token_dim), tuple(tokens.shape)
    tokens.square().mean().backward()
    assert feature_map.grad is not None
    assert float(feature_map.grad.abs().sum().item()) > 0.0
    print("grounded_token_check: PASS")


def run_random_shape_check(device: torch.device, batch: int, vector_dim: int, patch_radius: int) -> None:
    patch_size = 2 * patch_radius + 1
    vector = torch.randn(batch, vector_dim, device=device)
    patch = torch.randn(batch, 3, patch_size, patch_size, device=device)
    q_nominal = torch.randn(batch, 2, device=device)

    residual = ResidualCorrectionHead(vector_dim, patch_radius=patch_radius).to(device)
    delta_norm = residual(vector, patch)
    delta_px = residual.predict_delta_px(vector, patch)
    assert tuple(delta_norm.shape) == (batch, 2), tuple(delta_norm.shape)
    assert tuple(delta_px.shape) == (batch, 2), tuple(delta_px.shape)
    assert torch.allclose(delta_px, delta_norm * float(patch_radius))

    adapter = FCPGResidualOnly(vector_dim, patch_radius=patch_radius).to(device)
    output = adapter(vector, patch, q_nominal)
    assert tuple(output.delta_norm.shape) == (batch, 2)
    assert tuple(output.delta_px.shape) == (batch, 2)
    assert output.q_corrected_px is not None
    assert tuple(output.q_corrected_px.shape) == (batch, 2)
    assert torch.allclose(output.q_corrected_px, q_nominal + output.delta_px)

    reliability = ReliabilityConfidenceHead(vector_dim, pred_feature_dim=5, patch_radius=patch_radius).to(device)
    pred_features = torch.randn(batch, 5, device=device)
    rho = reliability(vector, patch, pred_features)
    assert tuple(rho.shape) == (batch, 1), tuple(rho.shape)
    assert bool(torch.all((rho >= 0.0) & (rho <= 1.0)).item())

    loss = output.delta_px.square().mean() + rho.mean()
    loss.backward()
    grad_norm = 0.0
    for param in list(adapter.parameters()) + list(reliability.parameters()):
        if param.grad is not None:
            grad_norm += float(param.grad.detach().abs().sum().item())
    assert grad_norm > 0.0
    print(f"random_shape_check: PASS batch={batch} vector_dim={vector_dim} patch_radius={patch_radius} device={device}")


def run_checkpoint_check(path: Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})
    vector_dim = int(checkpoint["vector_dim"])
    hidden_dim = int(args.get("hidden_dim", 128))
    patch_radius = int(args.get("patch_radius", 16))
    model = ResidualCorrectionHead(vector_dim, hidden_dim=hidden_dim, patch_radius=patch_radius).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    patch_size = 2 * patch_radius + 1
    vector = torch.randn(2, vector_dim, device=device)
    patch = torch.randn(2, 3, patch_size, patch_size, device=device)
    with torch.no_grad():
        delta_px = model.predict_delta_px(vector, patch)
    assert tuple(delta_px.shape) == (2, 2)
    print(f"checkpoint_check: PASS path={path} vector_dim={vector_dim} patch_radius={patch_radius}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--vector-dim", type=int, default=44)
    parser.add_argument("--patch-radius", type=int, default=16)
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()

    device = choose_device(args.device)
    run_feature_sampling_check(device)
    run_grounded_token_check(device)
    run_random_shape_check(device, args.batch, args.vector_dim, args.patch_radius)
    if args.checkpoint:
        run_checkpoint_check(Path(args.checkpoint), device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
