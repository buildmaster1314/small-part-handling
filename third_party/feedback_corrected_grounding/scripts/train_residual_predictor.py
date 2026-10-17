#!/usr/bin/env python3
"""Train no-image or local-image residual predictors on FCPG projection data."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.modules import ResidualCorrectionHead  # noqa: E402
from fcpg.observations import OBSERVATION_MODES, select_policy_observation  # noqa: E402


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    test: np.ndarray


class ProjectionResidualDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        indices: np.ndarray,
        patch_radius: int,
        include_obs: bool = True,
        require_inside_patch: bool = True,
        obs_mode: str = "full",
    ) -> None:
        self.path = Path(data_path)
        self.data = np.load(self.path, allow_pickle=False)
        self.rgb = self.data["rgb"]
        self.obs = select_policy_observation(self.data["obs"], obs_mode)
        self.keypoints_xyz = self.data["keypoints_xyz"]
        self.q_true = self.data["q_true"]
        self.q_nominal = self.data["q_nominal"]
        self.valid_true = self.data["valid_true"]
        self.valid_nominal = self.data["valid_nominal"]
        self.inside_patch = self.data["q_true_inside_patch"]
        self.patch_radius = int(patch_radius)
        self.include_obs = include_obs
        self.obs_mode = obs_mode
        self.height = int(self.rgb.shape[1])
        self.width = int(self.rgb.shape[2])
        self.num_keypoints = int(self.q_true.shape[1])

        pairs: list[tuple[int, int]] = []
        for sample_idx in indices.tolist():
            for keypoint_idx in range(self.num_keypoints):
                if not bool(self.valid_true[sample_idx, keypoint_idx]):
                    continue
                if not bool(self.valid_nominal[sample_idx, keypoint_idx]):
                    continue
                if require_inside_patch and not bool(self.inside_patch[sample_idx, keypoint_idx]):
                    continue
                pairs.append((sample_idx, keypoint_idx))
        if not pairs:
            raise RuntimeError("No valid sample/keypoint pairs after filtering")
        self.pairs = np.asarray(pairs, dtype=np.int64)
        self.vector_dim = self._make_vector(0, 0).shape[0]

    def __len__(self) -> int:
        return int(self.pairs.shape[0])

    def _make_vector(self, sample_idx: int, keypoint_idx: int) -> np.ndarray:
        q_nom = self.q_nominal[sample_idx, keypoint_idx].astype(np.float32)
        q_norm = np.array([q_nom[0] / self.width, q_nom[1] / self.height], dtype=np.float32)
        xyz = self.keypoints_xyz[sample_idx, keypoint_idx].astype(np.float32)
        onehot = np.zeros(self.num_keypoints, dtype=np.float32)
        onehot[keypoint_idx] = 1.0
        parts = [q_norm, xyz, onehot]
        if self.include_obs:
            parts.append(self.obs[sample_idx].astype(np.float32))
        return np.concatenate(parts, axis=0)

    def _crop_patch(self, sample_idx: int, keypoint_idx: int) -> np.ndarray:
        image = self.rgb[sample_idx]
        q_nom = self.q_nominal[sample_idx, keypoint_idx]
        cx = int(round(float(q_nom[0])))
        cy = int(round(float(q_nom[1])))
        radius = self.patch_radius
        size = 2 * radius + 1
        patch = np.zeros((size, size, 3), dtype=np.uint8)
        x0 = cx - radius
        y0 = cy - radius
        x1 = cx + radius + 1
        y1 = cy + radius + 1
        src_x0 = max(x0, 0)
        src_y0 = max(y0, 0)
        src_x1 = min(x1, self.width)
        src_y1 = min(y1, self.height)
        dst_x0 = src_x0 - x0
        dst_y0 = src_y0 - y0
        patch[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = image[
            src_y0:src_y1, src_x0:src_x1
        ]
        patch_f = patch.astype(np.float32) / 255.0
        return np.transpose(patch_f, (2, 0, 1))

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        sample_idx, keypoint_idx = self.pairs[item]
        target_px = (self.q_true[sample_idx, keypoint_idx] - self.q_nominal[sample_idx, keypoint_idx]).astype(np.float32)
        vector = self._make_vector(int(sample_idx), int(keypoint_idx))
        patch = self._crop_patch(int(sample_idx), int(keypoint_idx))
        target_norm_px = np.asarray([np.linalg.norm(target_px)], dtype=np.float32)
        return {
            "vector": torch.from_numpy(vector),
            "patch": torch.from_numpy(patch),
            "target": torch.from_numpy(target_px / float(self.patch_radius)),
            "target_px": torch.from_numpy(target_px),
            "target_norm_px": torch.from_numpy(target_norm_px),
        }


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def make_float_mask(values: np.ndarray, heldout_values: list[float], atol: float = 1e-4) -> np.ndarray:
    mask = np.zeros(values.shape, dtype=bool)
    for heldout in heldout_values:
        mask |= np.isclose(values.astype(np.float32), np.float32(heldout), atol=atol)
    return mask


def make_split(
    data_path: str | Path,
    heldout_directions: str,
    heldout_translations_cm: str,
    heldout_yaws_deg: str,
    val_fraction: float,
    seed: int,
    random_split_unit: str = "sample",
) -> SplitIndices:
    data = np.load(data_path, allow_pickle=False)
    n = int(data["rgb"].shape[0])
    all_indices = np.arange(n, dtype=np.int64)
    test_mask = np.zeros(n, dtype=bool)

    heldout_dirs = parse_csv_strings(heldout_directions)
    if heldout_dirs:
        directions = data["translation_direction"].astype(str)
        test_mask |= np.isin(directions, np.asarray(heldout_dirs))

    heldout_trans = parse_csv_floats(heldout_translations_cm)
    if heldout_trans:
        test_mask |= make_float_mask(data["translation_cm"], heldout_trans)

    heldout_yaws = parse_csv_floats(heldout_yaws_deg)
    if heldout_yaws:
        test_mask |= make_float_mask(data["yaw_deg"], heldout_yaws)

    if np.any(test_mask):
        train = all_indices[~test_mask]
        test = all_indices[test_mask]
        if len(train) > 0 and len(test) > 0:
            return SplitIndices(train=train, test=test)

    rng = np.random.default_rng(seed)
    if random_split_unit == "sample":
        shuffled = all_indices.copy()
        rng.shuffle(shuffled)
        test_size = max(1, int(round(n * val_fraction)))
        test = shuffled[:test_size]
        train = shuffled[test_size:]
        return SplitIndices(train=train, test=test)

    if random_split_unit == "task-episode":
        required = {"task_name", "episode_id"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"task-episode split requires dataset fields: {sorted(missing)}")
        task_names = data["task_name"].astype(str)
        episode_ids = data["episode_id"].astype(str)
        group_keys = np.char.add(np.char.add(task_names, ":"), episode_ids)
        selected_test_groups: list[str] = []
        for task_name in np.unique(task_names):
            task_groups = np.unique(group_keys[task_names == task_name])
            rng.shuffle(task_groups)
            test_group_count = max(1, int(round(len(task_groups) * val_fraction)))
            if test_group_count >= len(task_groups):
                test_group_count = len(task_groups) - 1
            if test_group_count <= 0:
                raise RuntimeError(f"task-episode split requires at least two episodes for {task_name}")
            selected_test_groups.extend(task_groups[:test_group_count].tolist())
        group_test_mask = np.isin(group_keys, np.asarray(selected_test_groups))
        return SplitIndices(train=all_indices[~group_test_mask], test=all_indices[group_test_mask])

    raise ValueError(f"Unknown random_split_unit: {random_split_unit}")


class NoImageResidualNet(nn.Module):
    def __init__(self, vector_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vector_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, vector: torch.Tensor, patch: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(vector)


class LocalImageResidualNet(ResidualCorrectionHead):
    """Backward-compatible alias for the reusable FCPG residual head."""




def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    patch_radius: int,
    clean_threshold_px: float = 1e-4,
) -> dict[str, float]:
    model.eval()
    losses = []
    before = []
    after = []
    clean_after = []
    clean_count = 0
    with torch.no_grad():
        for batch in loader:
            vector = batch["vector"].to(device)
            patch = batch["patch"].to(device)
            target = batch["target"].to(device)
            target_px = batch["target_px"].to(device)
            pred = model(vector, patch)
            losses.append(torch.nn.functional.smooth_l1_loss(pred, target).item())
            pred_px = pred * float(patch_radius)
            before_err = torch.linalg.norm(target_px, dim=1)
            after_err = torch.linalg.norm(pred_px - target_px, dim=1)
            clean_mask = before_err <= float(clean_threshold_px)
            if bool(clean_mask.any()):
                clean_after.append(after_err[clean_mask].detach().cpu())
                clean_count += int(clean_mask.sum().item())
            before.append(before_err.detach().cpu())
            after.append(after_err.detach().cpu())
    before_t = torch.cat(before)
    after_t = torch.cat(after)
    clean_after_t = torch.cat(clean_after) if clean_after else torch.empty(0)
    return {
        "loss": float(np.mean(losses)),
        "e_before": float(before_t.mean().item()),
        "e_after": float(after_t.mean().item()),
        "ratio": float((after_t.mean() / torch.clamp(before_t.mean(), min=1e-8)).item()),
        "e_after_p90": float(torch.quantile(after_t, 0.9).item()),
        "clean_count": int(clean_count),
        "clean_e_after": float(clean_after_t.mean().item()) if clean_after_t.numel() else float("nan"),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clean_loss_weight: float = 1.0,
    clean_threshold_px: float = 1e-4,
) -> float:
    model.train()
    losses = []
    for batch in loader:
        vector = batch["vector"].to(device)
        patch = batch["patch"].to(device)
        target = batch["target"].to(device)
        target_px = batch["target_px"].to(device)
        pred = model(vector, patch)
        per_sample = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none").mean(dim=1)
        if clean_loss_weight != 1.0:
            target_norm = torch.linalg.norm(target_px, dim=1)
            clean_mask = target_norm <= float(clean_threshold_px)
            weights = torch.ones_like(per_sample)
            weights = torch.where(clean_mask, weights * float(clean_loss_weight), weights)
            loss = (per_sample * weights).sum() / torch.clamp(weights.sum(), min=1e-8)
        else:
            loss = per_sample.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", choices=["no-image", "local-image"], required=True)
    parser.add_argument("--output-dir", default="outputs/residual_predictor")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--patch-radius", type=int, default=16)
    parser.add_argument("--heldout-directions", default="y-")
    parser.add_argument("--heldout-translations-cm", default="")
    parser.add_argument("--heldout-yaws-deg", default="")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--random-split-unit",
        choices=["sample", "task-episode"],
        default="sample",
        help="Use task-episode to keep all frames from one physical camera pose in one split.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-obs", action="store_true")
    parser.add_argument(
        "--obs-mode",
        choices=OBSERVATION_MODES,
        default="full",
        help="Select full legacy state or robot-only proprioception for residual inputs.",
    )
    parser.add_argument("--clean-loss-weight", type=float, default=1.0)
    parser.add_argument("--clean-threshold-px", type=float, default=1e-4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    split = make_split(
        args.data,
        args.heldout_directions,
        args.heldout_translations_cm,
        args.heldout_yaws_deg,
        args.val_fraction,
        args.seed,
        random_split_unit=args.random_split_unit,
    )
    train_ds = ProjectionResidualDataset(
        args.data,
        split.train,
        patch_radius=args.patch_radius,
        include_obs=not args.no_obs,
        require_inside_patch=True,
        obs_mode=args.obs_mode,
    )
    test_ds = ProjectionResidualDataset(
        args.data,
        split.test,
        patch_radius=args.patch_radius,
        include_obs=not args.no_obs,
        require_inside_patch=True,
        obs_mode=args.obs_mode,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.model == "no-image":
        model = NoImageResidualNet(train_ds.vector_dim, hidden_dim=args.hidden_dim)
    else:
        model = LocalImageResidualNet(train_ds.vector_dim, hidden_dim=args.hidden_dim, patch_radius=args.patch_radius)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = []
    best_metrics = None
    best_state = None
    print(f"data: {args.data}")
    print(f"model: {args.model}")
    print(f"device: {device}")
    print(f"random_split_unit: {args.random_split_unit}")
    print(f"train_samples: {len(train_ds)} pairs from {len(split.train)} frames")
    print(f"test_samples: {len(test_ds)} pairs from {len(split.test)} frames")
    print(f"vector_dim: {train_ds.vector_dim}")
    print(f"obs_mode: {args.obs_mode}")
    print(f"clean_loss_weight: {args.clean_loss_weight} clean_threshold_px: {args.clean_threshold_px}")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            clean_loss_weight=args.clean_loss_weight,
            clean_threshold_px=args.clean_threshold_px,
        )
        metrics = evaluate(model, test_loader, device, args.patch_radius, clean_threshold_px=args.clean_threshold_px)
        metrics = {"epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(metrics)
        if best_metrics is None or metrics["ratio"] < best_metrics["ratio"]:
            best_metrics = dict(metrics)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"epoch {epoch:03d} train_loss={train_loss:.5f} "
            f"test_e_before={metrics['e_before']:.4f} "
            f"test_e_after={metrics['e_after']:.4f} "
            f"ratio={metrics['ratio']:.4f} "
            f"clean_e_after={metrics['clean_e_after']:.4f} clean_n={metrics['clean_count']}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{args.model}_residual.pt"
    best_model_path = output_dir / f"{args.model}_residual_best.pt"
    metrics_path = output_dir / f"{args.model}_metrics.json"
    torch.save({"model_state": model.state_dict(), "args": vars(args), "vector_dim": train_ds.vector_dim}, model_path)
    if best_state is not None:
        torch.save({"model_state": best_state, "args": vars(args), "vector_dim": train_ds.vector_dim}, best_model_path)
    final = history[-1]
    metrics_path.write_text(json.dumps({"final": final, "best": best_metrics, "history": history}, indent=2), encoding="utf-8")
    print(f"model_path: {model_path}")
    print(f"best_model_path: {best_model_path}")
    print(f"metrics_path: {metrics_path}")
    if best_metrics is not None:
        print(f"best_epoch: {best_metrics['epoch']} best_ratio: {best_metrics['ratio']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
