#!/usr/bin/env python3
"""Train a reliability head for grounded-token confidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.modules import ReliabilityConfidenceHead  # noqa: E402
from scripts.train_residual_predictor import (  # noqa: E402
    LocalImageResidualNet,
    ProjectionResidualDataset,
    make_split,
)


class ReliabilityDataset(Dataset):
    def __init__(
        self,
        base_dataset: ProjectionResidualDataset,
        residual_model: nn.Module,
        device: torch.device,
        sigma_px: float,
        target_kind: str,
        include_pred_features: bool,
    ) -> None:
        self.base = base_dataset
        self.device = device
        self.sigma_px = float(sigma_px)
        self.target_kind = target_kind
        self.include_pred_features = bool(include_pred_features)
        self.pred_feature_dim = 5 if self.include_pred_features else 0
        self.targets = []
        self.before_errors = []
        self.corrected_errors = []
        self.pred_features = []
        residual_model.eval()
        loader = DataLoader(base_dataset, batch_size=256, shuffle=False, num_workers=0)
        with torch.no_grad():
            for batch in loader:
                vector = batch["vector"].to(device)
                patch = batch["patch"].to(device)
                target_px = batch["target_px"].to(device)
                pred_px = residual_model(vector, patch) * float(base_dataset.patch_radius)
                before_err = torch.linalg.norm(target_px, dim=1)
                after_err = torch.linalg.norm(pred_px - target_px, dim=1)
                if target_kind == "corr":
                    target_err = after_err
                elif target_kind == "nom":
                    target_err = before_err
                else:
                    raise ValueError(f"unsupported target_kind: {target_kind}")
                rho = torch.exp(-target_err / max(self.sigma_px, 1e-6))
                self.before_errors.append(before_err.cpu())
                self.corrected_errors.append(after_err.cpu())
                self.targets.append(rho.cpu())
                if self.include_pred_features:
                    radius = float(base_dataset.patch_radius)
                    pred_norm = torch.linalg.norm(pred_px, dim=1, keepdim=True)
                    feat = torch.cat([pred_px / radius, pred_norm / radius, pred_px.abs() / radius], dim=1)
                    self.pred_features.append(feat.cpu())
        self.targets_t = torch.cat(self.targets).float()
        self.before_errors_t = torch.cat(self.before_errors).float()
        self.errors_t = torch.cat(self.corrected_errors).float()
        if self.include_pred_features:
            self.pred_features_t = torch.cat(self.pred_features).float()
        else:
            self.pred_features_t = torch.zeros((len(self.base), 0), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.base[index]
        item["rho_target"] = self.targets_t[index].unsqueeze(0)
        item["before_error"] = self.before_errors_t[index].unsqueeze(0)
        item["corrected_error"] = self.errors_t[index].unsqueeze(0)
        item["pred_features"] = self.pred_features_t[index]
        return item


class ReliabilityHead(ReliabilityConfidenceHead):
    """Backward-compatible alias for the reusable reliability head."""




def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_residual_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})
    vector_dim = int(checkpoint["vector_dim"])
    hidden_dim = int(args.get("hidden_dim", 128))
    patch_radius = int(args.get("patch_radius", 16))
    model = LocalImageResidualNet(vector_dim, hidden_dim=hidden_dim, patch_radius=patch_radius)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, args, vector_dim, patch_radius


def pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    if float(denom) < 1e-8:
        return 0.0
    return float((a @ b / denom).item())


def model_inputs(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return batch["vector"].to(device), batch["patch"].to(device), batch["pred_features"].to(device)


def confidence_loss(pred: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "mse":
        return torch.nn.functional.mse_loss(pred, target)
    if loss_name == "bce":
        eps = 1e-5
        return torch.nn.functional.binary_cross_entropy(pred.clamp(eps, 1.0 - eps), target)
    raise ValueError(f"unsupported loss: {loss_name}")


def evaluate_reliability(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_name: str,
) -> dict[str, float]:
    model.eval()
    preds = []
    targets = []
    before_errors = []
    errors = []
    losses = []
    with torch.no_grad():
        for batch in loader:
            vector, patch, pred_features = model_inputs(batch, device)
            target = batch["rho_target"].to(device)
            pred = model(vector, patch, pred_features)
            losses.append(confidence_loss(pred, target, loss_name).item())
            preds.append(pred.cpu())
            targets.append(target.cpu())
            before_errors.append(batch["before_error"].cpu())
            errors.append(batch["corrected_error"].cpu())
    pred_t = torch.cat(preds).flatten()
    target_t = torch.cat(targets).flatten()
    before_t = torch.cat(before_errors).flatten()
    error_t = torch.cat(errors).flatten()
    corr_target = pearson_corr(pred_t, target_t)
    corr_error = pearson_corr(pred_t, -error_t)
    k = max(1, int(0.25 * len(pred_t)))
    order = torch.argsort(pred_t)
    low_idx = order[:k]
    high_idx = order[-k:]
    return {
        "loss": float(np.mean(losses)),
        "pred_mean": float(pred_t.mean().item()),
        "pred_std": float(pred_t.std(unbiased=False).item()),
        "target_mean": float(target_t.mean().item()),
        "target_std": float(target_t.std(unbiased=False).item()),
        "before_error_mean": float(before_t.mean().item()),
        "corrected_error_mean": float(error_t.mean().item()),
        "corr_pred_target": corr_target,
        "corr_pred_neg_error": corr_error,
        "high_conf_error": float(error_t[high_idx].mean().item()),
        "low_conf_error": float(error_t[low_idx].mean().item()),
        "high_conf_target": float(target_t[high_idx].mean().item()),
        "low_conf_target": float(target_t[low_idx].mean().item()),
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_name: str,
) -> float:
    model.train()
    losses = []
    for batch in loader:
        vector, patch, pred_features = model_inputs(batch, device)
        target = batch["rho_target"].to(device)
        pred = model(vector, patch, pred_features)
        loss = confidence_loss(pred, target, loss_name)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--residual-checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/reliability_head")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--sigma-px", type=float, default=1.0)
    parser.add_argument("--target-kind", choices=["corr", "nom"], default="corr")
    parser.add_argument("--loss", choices=["mse", "bce"], default="bce")
    parser.add_argument("--no-pred-features", action="store_true")
    parser.add_argument("--heldout-directions", default="")
    parser.add_argument("--heldout-translations-cm", default="4")
    parser.add_argument("--heldout-yaws-deg", default="")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)

    residual_model, residual_args, vector_dim, patch_radius = load_residual_model(Path(args.residual_checkpoint), device)
    include_obs = not bool(residual_args.get("no_obs", False))
    residual_obs_mode = str(residual_args.get("obs_mode", "full"))
    split = make_split(
        args.data,
        args.heldout_directions,
        args.heldout_translations_cm,
        args.heldout_yaws_deg,
        args.val_fraction,
        args.seed,
    )
    include_pred_features = not args.no_pred_features
    train_base = ProjectionResidualDataset(args.data, split.train, patch_radius=patch_radius, include_obs=include_obs, obs_mode=residual_obs_mode)
    test_base = ProjectionResidualDataset(args.data, split.test, patch_radius=patch_radius, include_obs=include_obs, obs_mode=residual_obs_mode)
    train_ds = ReliabilityDataset(
        train_base,
        residual_model,
        device,
        sigma_px=args.sigma_px,
        target_kind=args.target_kind,
        include_pred_features=include_pred_features,
    )
    test_ds = ReliabilityDataset(
        test_base,
        residual_model,
        device,
        sigma_px=args.sigma_px,
        target_kind=args.target_kind,
        include_pred_features=include_pred_features,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ReliabilityHead(
        vector_dim,
        pred_feature_dim=train_ds.pred_feature_dim,
        hidden_dim=args.hidden_dim,
        patch_radius=patch_radius,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    best = None
    best_state = None
    print(f"device: {device}")
    print(f"train_pairs: {len(train_ds)} test_pairs: {len(test_ds)}")
    print(f"sigma_px: {args.sigma_px} target_kind: {args.target_kind} loss: {args.loss}")
    print(f"pred_feature_dim: {train_ds.pred_feature_dim}")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, args.loss)
        metrics = evaluate_reliability(model, test_loader, device, args.loss)
        metrics = {"epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(metrics)
        if best is None or metrics["corr_pred_neg_error"] > best["corr_pred_neg_error"]:
            best = dict(metrics)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"epoch {epoch:03d} train_loss={train_loss:.5f} "
            f"test_loss={metrics['loss']:.5f} "
            f"pred_mean={metrics['pred_mean']:.4f} target_mean={metrics['target_mean']:.4f} "
            f"corr_pred_target={metrics['corr_pred_target']:.4f} "
            f"corr_pred_neg_error={metrics['corr_pred_neg_error']:.4f} "
            f"high_err={metrics['high_conf_error']:.4f} low_err={metrics['low_conf_error']:.4f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "reliability_head.pt"
    best_model_path = output_dir / "reliability_head_best.pt"
    metrics_path = output_dir / "reliability_metrics.json"
    payload = {
        "model_state": model.state_dict(),
        "args": vars(args),
        "vector_dim": vector_dim,
        "pred_feature_dim": train_ds.pred_feature_dim,
        "patch_radius": patch_radius,
    }
    torch.save(payload, model_path)
    if best_state is not None:
        best_payload = dict(payload)
        best_payload["model_state"] = best_state
        torch.save(best_payload, best_model_path)
    metrics_path.write_text(json.dumps({"final": history[-1], "best": best, "history": history}, indent=2), encoding="utf-8")
    print(f"model_path: {model_path}")
    print(f"best_model_path: {best_model_path}")
    print(f"metrics_path: {metrics_path}")
    if best:
        print(f"best_epoch: {best['epoch']} corr_pred_neg_error: {best['corr_pred_neg_error']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
