#!/usr/bin/env python3
"""Train legacy or LeWM-backed FCPG projection correctors.

This stage is policy-free. It tests whether anchor-local visual evidence can
predict the geometric projection residual before any action policy is trained.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.observations import OBSERVATION_MODES, select_policy_observation  # noqa: E402
from fcpg.lewm_core import LEWM_UPSTREAM_COMMIT  # noqa: E402
from fcpg.regularization import SIGReg  # noqa: E402
from fcpg.sequence_data import (  # noqa: E402
    HomogeneousConcatDataset,
    make_balanced_source_sampler,
    resolve_projection_sources,
    select_episode_subset,
)
from fcpg.transformer_grounding import (  # noqa: E402
    DEFAULT_VIT_MODEL,
    FCPGViTAdapter,
    LeWMFCPGAdapter,
)
from scripts.train_bc_policy import (  # noqa: E402
    SplitIndices,
    make_split,
    require_pre_action_alignment,
)


CORRECTOR_ARCHITECTURES = ("legacy", "lewm")
CHECKPOINT_SCHEMA_VERSION = 2


class TransformerResidualDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        indices: np.ndarray,
        obs_mode: str,
        vision_cache: str | Path | None = None,
    ) -> None:
        self.path = Path(data_path)
        self.data = np.load(self.path, allow_pickle=False)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.vision_cache_path = Path(vision_cache) if vision_cache else None
        self.vision_backbone_features = None
        self.vision_prefix = None
        if self.vision_cache_path is None:
            self.rgb = self.data["rgb"]
            self.image_size = (int(self.rgb.shape[2]), int(self.rgb.shape[1]))
        else:
            metadata = json.loads(
                (self.vision_cache_path / "metadata.json").read_text(encoding="utf-8")
            )
            source_data = metadata.get("source_data")
            if source_data and Path(source_data).resolve() != self.path.resolve():
                raise ValueError(
                    f"vision cache source {source_data} does not match dataset {self.path.resolve()}"
                )
            self.rgb = None
            self.image_size = (int(metadata["image_width"]), int(metadata["image_height"]))
            self.vision_backbone_features = np.load(
                self.vision_cache_path / "backbone_features.npy",
                mmap_mode="r",
            )
            self.vision_prefix = np.load(
                self.vision_cache_path / "prefix.npy",
                mmap_mode="r",
            )
        self.obs = select_policy_observation(self.data["obs"], obs_mode)
        self.keypoints_xyz = self.data["keypoints_xyz"].astype(np.float32)
        self.q_nominal = self.data["q_nominal"].astype(np.float32)
        self.q_true = self.data["q_true"].astype(np.float32)
        self.valid = (self.data["valid_true"] & self.data["valid_nominal"]).astype(bool)
        self.obs_dim = int(self.obs.shape[-1])
        self.num_keypoints = int(self.q_nominal.shape[1])
        if self.vision_backbone_features is not None:
            if self.vision_backbone_features.shape[0] != self.obs.shape[0]:
                raise ValueError("vision cache sample count does not match residual dataset")
            if self.vision_prefix is None or self.vision_prefix.shape[0] != self.obs.shape[0]:
                raise ValueError("vision cache prefix count does not match residual dataset")

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[item])
        result = {
            "obs": torch.from_numpy(self.obs[idx]),
            "keypoints_xyz": torch.from_numpy(self.keypoints_xyz[idx]),
            "q_nominal": torch.from_numpy(self.q_nominal[idx]),
            "q_true": torch.from_numpy(self.q_true[idx]),
            "valid": torch.from_numpy(self.valid[idx]),
            "image_size": torch.tensor(self.image_size, dtype=torch.long),
        }
        if self.rgb is not None:
            rgb = np.transpose(self.rgb[idx].astype(np.float32) / 255.0, (2, 0, 1))
            result["rgb"] = torch.from_numpy(rgb)
        else:
            assert self.vision_backbone_features is not None
            assert self.vision_prefix is not None
            result["vision_backbone_features"] = torch.from_numpy(
                np.array(self.vision_backbone_features[idx], dtype=np.float16, copy=True)
            )
            result["vision_prefix"] = torch.from_numpy(
                np.array(self.vision_prefix[idx], dtype=np.float16, copy=True)
            )
        return result


def parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not layers:
        raise argparse.ArgumentTypeError("at least one ViT layer is required")
    return layers


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in batch.items()
    }


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(dtype=value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def corrupt_visual_evidence(
    rgb: torch.Tensor,
    q_nominal: torch.Tensor,
    probability: float,
    cutout_radius_px: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create correspondence-breaking negatives for reliability supervision."""

    batch, _, height, width = rgb.shape
    corrupted = torch.rand(batch, device=rgb.device) < float(probability)
    if not bool(corrupted.any()):
        return rgb, corrupted
    use_shuffle = corrupted & (torch.rand(batch, device=rgb.device) < 0.5)
    use_cutout = corrupted & ~use_shuffle
    result = rgb.clone()
    if bool(use_shuffle.any()):
        permutation = torch.randperm(batch, device=rgb.device)
        result[use_shuffle] = rgb[permutation[use_shuffle]]
    if bool(use_cutout.any()):
        ys = torch.arange(height, device=rgb.device, dtype=q_nominal.dtype)
        xs = torch.arange(width, device=rgb.device, dtype=q_nominal.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        dx = (xx[None, None] - q_nominal[..., 0, None, None]).abs()
        dy = (yy[None, None] - q_nominal[..., 1, None, None]).abs()
        local_mask = ((dx <= cutout_radius_px) & (dy <= cutout_radius_px)).any(dim=1)
        local_mask = local_mask & use_cutout[:, None, None]
        fill = torch.rand(batch, rgb.shape[1], 1, 1, device=rgb.device, dtype=rgb.dtype)
        result = torch.where(local_mask[:, None], fill, result)
    return result, corrupted


def corrupt_cached_visual_evidence(
    feature_maps: torch.Tensor,
    prefix: torch.Tensor,
    q_nominal: torch.Tensor,
    image_size: tuple[int, int],
    probability: float,
    cutout_radius_px: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create shuffled-image and anchor-local cutout negatives in ViT space."""

    batch, _, _, feature_height, feature_width = feature_maps.shape
    corrupted = torch.rand(batch, device=feature_maps.device) < float(probability)
    if not bool(corrupted.any()):
        return feature_maps, prefix, corrupted
    use_shuffle = corrupted & (torch.rand(batch, device=feature_maps.device) < 0.5)
    use_cutout = corrupted & ~use_shuffle
    result_features = feature_maps.clone()
    result_prefix = prefix.clone()
    if bool(use_shuffle.any()):
        permutation = torch.randperm(batch, device=feature_maps.device)
        result_features[use_shuffle] = feature_maps[permutation[use_shuffle]]
        result_prefix[use_shuffle] = prefix[permutation[use_shuffle]]
    if bool(use_cutout.any()):
        image_width, image_height = image_size
        xs = (torch.arange(feature_width, device=feature_maps.device, dtype=q_nominal.dtype) + 0.5)
        ys = (torch.arange(feature_height, device=feature_maps.device, dtype=q_nominal.dtype) + 0.5)
        xs = xs * (float(image_width) / float(feature_width))
        ys = ys * (float(image_height) / float(feature_height))
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        dx = (xx[None, None] - q_nominal[..., 0, None, None]).abs()
        dy = (yy[None, None] - q_nominal[..., 1, None, None]).abs()
        local_mask = ((dx <= cutout_radius_px) & (dy <= cutout_radius_px)).any(dim=1)
        local_mask = local_mask & use_cutout[:, None, None]
        result_features = torch.where(
            local_mask[:, None, None],
            torch.zeros_like(result_features),
            result_features,
        )
    return result_features, result_prefix, corrupted


def compute_losses(
    output,
    batch: dict[str, torch.Tensor],
    reliability_sigma_px: float,
    reliability_weight: float,
    sigreg: SIGReg | None,
    sigreg_weight: float,
    corrupted: torch.Tensor | None = None,
    corrector_arch: str = "legacy",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if corrector_arch not in {"legacy", "lewm"}:
        raise ValueError(f"unknown corrector architecture: {corrector_arch}")
    if corrector_arch == "lewm" and corrupted is not None and bool(corrupted.any()):
        raise ValueError("LeWM residual-only training cannot use reliability corruptions")

    target_delta = batch["q_true"] - batch["q_nominal"]
    valid = batch["valid"]
    residual_per_keypoint = F.smooth_l1_loss(
        output.delta_px,
        target_delta,
        reduction="none",
        beta=1.0,
    ).mean(dim=-1)
    residual_valid = valid if corrupted is None else valid & ~corrupted[:, None]
    residual_loss = masked_mean(residual_per_keypoint, residual_valid)
    corrected_error = torch.linalg.vector_norm(
        output.q_corrected_px - batch["q_true"], dim=-1
    )

    if sigreg is None or sigreg_weight <= 0:
        sigreg_loss = residual_loss.new_zeros(())
    else:
        sigreg_loss = sigreg(output.keypoint_latent.transpose(0, 1).contiguous())

    parts = {
        "residual_loss": residual_loss,
        "sigreg_loss": sigreg_loss,
        "corrected_error": masked_mean(corrected_error, residual_valid),
    }
    total = residual_loss + float(sigreg_weight) * sigreg_loss
    if corrector_arch == "lewm":
        return total, parts

    if not hasattr(output, "reliability_logit") or not hasattr(output, "reliability"):
        raise TypeError("legacy corrector output must include reliability fields")
    reliability_target = valid.to(dtype=corrected_error.dtype) * torch.exp(
        -corrected_error.detach() / float(reliability_sigma_px)
    )
    if corrupted is not None:
        reliability_target = torch.where(
            corrupted[:, None],
            torch.zeros_like(reliability_target),
            reliability_target,
        )
    reliability_per_keypoint = F.binary_cross_entropy_with_logits(
        output.reliability_logit,
        reliability_target,
        reduction="none",
    )
    reliability_loss = masked_mean(reliability_per_keypoint, valid)
    total = total + float(reliability_weight) * reliability_loss
    parts.update(
        {
            "reliability_loss": reliability_loss,
            "reliability": masked_mean(output.reliability, valid),
        }
    )
    return total, parts


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    reliability_sigma_px: float,
    reliability_weight: float,
    sigreg: SIGReg | None,
    sigreg_weight: float,
    grad_clip: float,
    reliability_corruption_prob: float,
    cutout_radius_px: float,
    corrector_arch: str = "legacy",
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "residual_loss": 0.0,
        "sigreg_loss": 0.0,
        "e_before": 0.0,
        "e_after": 0.0,
    }
    if corrector_arch == "legacy":
        totals.update(
            {
                "reliability_loss": 0.0,
                "reliability": 0.0,
            }
        )
    count = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            batch = move_batch(batch, device)
            corrupted = None
            amp_enabled = device.type == "cuda"
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                if "rgb" in batch:
                    rgb = batch["rgb"]
                    if (
                        corrector_arch == "legacy"
                        and training
                        and reliability_corruption_prob > 0
                    ):
                        rgb, corrupted = corrupt_visual_evidence(
                            rgb,
                            batch["q_nominal"],
                            reliability_corruption_prob,
                            cutout_radius_px,
                        )
                    output = model(
                        rgb,
                        batch["q_nominal"],
                        batch["keypoints_xyz"],
                        batch["obs"],
                    )
                else:
                    feature_maps = batch["vision_backbone_features"]
                    prefix = batch["vision_prefix"]
                    image_sizes = batch["image_size"]
                    if not bool(torch.all(image_sizes == image_sizes[0])):
                        raise ValueError("all cached residual samples must share one image size")
                    image_size = tuple(int(value) for value in image_sizes[0].tolist())
                    if (
                        corrector_arch == "legacy"
                        and training
                        and reliability_corruption_prob > 0
                    ):
                        feature_maps, prefix, corrupted = corrupt_cached_visual_evidence(
                            feature_maps,
                            prefix,
                            batch["q_nominal"],
                            image_size,
                            reliability_corruption_prob,
                            cutout_radius_px,
                        )
                    output = model.forward_from_backbone_features(
                        feature_maps,
                        prefix,
                        batch["q_nominal"],
                        batch["keypoints_xyz"],
                        batch["obs"],
                        image_size=image_size,
                    )
                loss, parts = compute_losses(
                    output,
                    batch,
                    reliability_sigma_px=reliability_sigma_px,
                    reliability_weight=reliability_weight,
                    sigreg=sigreg,
                    sigreg_weight=sigreg_weight,
                    corrupted=corrupted,
                    corrector_arch=corrector_arch,
                )
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    max_norm=grad_clip,
                )
                scaler.step(optimizer)
                scaler.update()

            valid = batch["valid"]
            e_before = masked_mean(
                torch.linalg.vector_norm(batch["q_nominal"] - batch["q_true"], dim=-1),
                valid,
            )
            batch_size = int(batch["obs"].shape[0])
            totals["loss"] += float(loss.detach()) * batch_size
            totals["residual_loss"] += float(parts["residual_loss"].detach()) * batch_size
            totals["sigreg_loss"] += float(parts["sigreg_loss"].detach()) * batch_size
            totals["e_before"] += float(e_before.detach()) * batch_size
            totals["e_after"] += float(parts["corrected_error"].detach()) * batch_size
            if corrector_arch == "legacy":
                totals["reliability_loss"] += (
                    float(parts["reliability_loss"].detach()) * batch_size
                )
                totals["reliability"] += float(parts["reliability"].detach()) * batch_size
            count += batch_size
    metrics = {key: value / max(count, 1) for key, value in totals.items()}
    metrics["ratio"] = metrics["e_after"] / max(metrics["e_before"], 1e-8)
    return metrics


def cosine_warmup_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(warmup_steps, 1))
    progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_json_write(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def resolve_training_weights(
    corrector_arch: str,
    reliability_weight: float | None,
    reliability_corruption_prob: float | None,
    sigreg_weight: float | None,
) -> tuple[float, float, float]:
    if corrector_arch not in CORRECTOR_ARCHITECTURES:
        raise ValueError(f"unknown corrector architecture: {corrector_arch}")
    if corrector_arch == "legacy":
        resolved_reliability = 0.2 if reliability_weight is None else float(reliability_weight)
        resolved_corruption = (
            0.25
            if reliability_corruption_prob is None
            else float(reliability_corruption_prob)
        )
        resolved_sigreg = 1e-4 if sigreg_weight is None else float(sigreg_weight)
    else:
        resolved_reliability = 0.0 if reliability_weight is None else float(reliability_weight)
        resolved_corruption = (
            0.0
            if reliability_corruption_prob is None
            else float(reliability_corruption_prob)
        )
        resolved_sigreg = 0.02 if sigreg_weight is None else float(sigreg_weight)
        if resolved_reliability != 0.0:
            raise ValueError("LeWM residual-only corrector requires --reliability-weight 0")
        if resolved_corruption != 0.0:
            raise ValueError(
                "LeWM residual-only corrector requires --reliability-corruption-prob 0"
            )
        if not (
            math.isclose(resolved_sigreg, 0.0, abs_tol=1e-12)
            or math.isclose(resolved_sigreg, 0.02, rel_tol=1e-9, abs_tol=1e-12)
            or math.isclose(resolved_sigreg, 0.09, rel_tol=1e-9, abs_tol=1e-12)
        ):
            raise ValueError(
                "LeWM corrector supports matched SIGReg weights 0, 0.02, or 0.09"
            )

    if resolved_reliability < 0 or resolved_sigreg < 0:
        raise ValueError("loss weights must be non-negative")
    if not 0.0 <= resolved_corruption <= 1.0:
        raise ValueError("reliability corruption probability must be in [0, 1]")
    return resolved_reliability, resolved_corruption, resolved_sigreg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--corrector-arch", choices=CORRECTOR_ARCHITECTURES, default="legacy"
    )
    parser.add_argument("--obs-mode", choices=OBSERVATION_MODES, default="robot")
    parser.add_argument("--model-name", default=DEFAULT_VIT_MODEL)
    parser.add_argument("--vision-cache", default="")
    parser.add_argument("--extra-data", action="append", default=[])
    parser.add_argument("--extra-vision-cache", action="append", default=[])
    parser.add_argument(
        "--balance-sources",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--vision-selected-layers", type=parse_layers, default=(2, 5, 8, 11))
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--vision-dim", type=int, default=256)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--local-grid-size", type=int, default=5)
    parser.add_argument("--local-radius-px", type=float, default=16.0)
    parser.add_argument("--local-layers", type=int, default=4)
    parser.add_argument("--global-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-residual-px", type=float, default=20.0)
    parser.add_argument("--reliability-sigma-px", type=float, default=3.0)
    parser.add_argument("--reliability-weight", type=float, default=None)
    parser.add_argument("--reliability-corruption-prob", type=float, default=None)
    parser.add_argument("--cutout-radius-px", type=float, default=12.0)
    parser.add_argument("--sigreg-weight", type=float, default=None)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--sigreg-proj", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-unit", choices=["sample", "task-episode"], default="task-episode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--max-train-episodes-per-source", type=int, default=0)
    parser.add_argument("--episode-selection-seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-unaligned-data", action="store_true")
    args = parser.parse_args()
    try:
        (
            args.reliability_weight,
            args.reliability_corruption_prob,
            args.sigreg_weight,
        ) = resolve_training_weights(
            args.corrector_arch,
            args.reliability_weight,
            args.reliability_corruption_prob,
            args.sigreg_weight,
        )
    except ValueError as error:
        parser.error(str(error))

    try:
        sources = resolve_projection_sources(
            args.data,
            args.vision_cache or None,
            args.extra_data,
            args.extra_vision_cache,
        )
    except ValueError as error:
        parser.error(str(error))
    if any(source.vision_cache is not None for source in sources) and args.unfreeze_backbone:
        parser.error("--vision-cache requires a frozen ViT backbone")
    for source in sources:
        if source.vision_cache is None:
            continue
        cache_metadata = json.loads(
            (source.vision_cache / "metadata.json").read_text(encoding="utf-8")
        )
        source_data = cache_metadata.get("source_data")
        if source_data and Path(source_data).resolve() != source.data.resolve():
            parser.error(
                f"vision cache source {source_data} does not match dataset "
                f"{source.data.resolve()}"
            )
        cached_layers = tuple(int(layer) for layer in cache_metadata["selected_layers"])
        if cached_layers != args.vision_selected_layers:
            parser.error(
                f"vision cache layers {cached_layers} do not match "
                f"--vision-selected-layers {args.vision_selected_layers}"
            )
        if str(cache_metadata["model_name"]) != args.model_name:
            parser.error("vision cache model does not match --model-name")
    if args.batch_size <= 0 or args.epochs <= 0:
        parser.error("--batch-size and --epochs must be positive")
    if args.max_train_episodes_per_source < 0:
        parser.error("--max-train-episodes-per-source must be non-negative")
    if not 0.0 <= args.reliability_corruption_prob <= 1.0:
        parser.error("--reliability-corruption-prob must be in [0, 1]")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    source_splits = []
    alignments = []
    for source in sources:
        with np.load(source.data, allow_pickle=False) as raw:
            alignments.append(
                require_pre_action_alignment(
                    raw,
                    source.data,
                    args.allow_unaligned_data,
                )
            )
            split = make_split(raw, args.val_fraction, args.split_seed, args.split_unit)
            if args.max_train_episodes_per_source > 0:
                split = SplitIndices(
                    train=select_episode_subset(
                        raw,
                        split.train,
                        args.max_train_episodes_per_source,
                        args.episode_selection_seed,
                    ),
                    test=split.test,
                )
            source_splits.append(split)
    train_sources = [
        TransformerResidualDataset(
            source.data,
            split.train,
            args.obs_mode,
            vision_cache=source.vision_cache,
        )
        for source, split in zip(sources, source_splits, strict=True)
    ]
    val_sources = [
        TransformerResidualDataset(
            source.data,
            split.test,
            args.obs_mode,
            vision_cache=source.vision_cache,
        )
        for source, split in zip(sources, source_splits, strict=True)
    ]
    if len(train_sources) == 1:
        train_dataset = train_sources[0]
        val_dataset = val_sources[0]
    else:
        shared_attributes = ("obs_dim", "num_keypoints", "image_size")
        train_dataset = HomogeneousConcatDataset(train_sources, shared_attributes)
        val_dataset = HomogeneousConcatDataset(val_sources, shared_attributes)
    train_sampler = (
        make_balanced_source_sampler(train_sources, args.seed)
        if len(train_sources) > 1 and args.balance_sources
        else None
    )
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    model_class = FCPGViTAdapter if args.corrector_arch == "legacy" else LeWMFCPGAdapter
    model = model_class(
        obs_dim=train_dataset.obs_dim,
        num_keypoints=train_dataset.num_keypoints,
        vision_dim=args.vision_dim,
        model_dim=args.model_dim,
        model_name=args.model_name,
        vision_selected_layers=args.vision_selected_layers,
        pretrained=not args.no_pretrained,
        freeze_backbone=not args.unfreeze_backbone,
        local_grid_size=args.local_grid_size,
        local_radius_px=args.local_radius_px,
        local_layers=args.local_layers,
        global_layers=args.global_layers,
        heads=args.heads,
        dropout=args.dropout,
        max_residual_px=args.max_residual_px,
    ).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        initial_args = dict(initial.get("args", {}))
        initial_arch = str(
            initial.get("corrector_arch", initial_args.get("corrector_arch", "legacy"))
        ).lower()
        if initial_arch != args.corrector_arch:
            parser.error(
                f"init checkpoint architecture {initial_arch} does not match "
                f"--corrector-arch {args.corrector_arch}"
            )
        model.load_state_dict(initial["model_state"])
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        betas=(0.95, 0.999),
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, args.epochs)
    warmup_steps = int(round(args.warmup_epochs))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup_lambda(step, warmup_steps, total_steps),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    sigreg = SIGReg(args.sigreg_knots, args.sigreg_proj).to(device) if args.sigreg_weight > 0 else None

    source_manifest = [
        {
            "data": str(source.data.resolve()),
            "vision_cache": (
                str(source.vision_cache.resolve())
                if source.vision_cache is not None
                else None
            ),
            "alignment": alignment,
            "train_frames": int(len(train_source)),
            "val_frames": int(len(val_source)),
        }
        for source, alignment, train_source, val_source in zip(
            sources,
            alignments,
            train_sources,
            val_sources,
            strict=True,
        )
    ]
    print(f"sources: {json.dumps(source_manifest)}")
    print(f"balanced_sources: {bool(train_sampler is not None)}")
    print(f"device: {device}")
    print(f"obs_mode: {args.obs_mode}")
    print(f"corrector_arch: {args.corrector_arch}")
    print(
        f"loss_weights: reliability={args.reliability_weight} "
        f"sigreg={args.sigreg_weight} corruption={args.reliability_corruption_prob}"
    )
    print(f"model_name: {args.model_name} pretrained: {not args.no_pretrained}")
    print(
        f"vision_layers: {args.vision_selected_layers} "
        f"caches: {[item['vision_cache'] for item in source_manifest]}"
    )
    print(f"backbone_frozen: {not args.unfreeze_backbone}")
    print(f"train_frames: {len(train_dataset)} val_frames: {len(val_dataset)}")
    print(f"batch_size: {args.batch_size} num_workers: {args.num_workers}")
    print(f"parameters_total: {sum(p.numel() for p in model.parameters())}")
    print(f"parameters_trainable: {sum(p.numel() for p in trainable)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "fcpg_transformer_residual.pt"
    best_path = output_dir / "fcpg_transformer_residual_best.pt"
    metrics_path = output_dir / "metrics.json"
    loss_config = {
        "residual": "smooth_l1_beta_1",
        "reliability_weight": args.reliability_weight,
        "reliability_corruption_prob": args.reliability_corruption_prob,
        "sigreg_weight": args.sigreg_weight,
        "sigreg_knots": args.sigreg_knots,
        "sigreg_proj": args.sigreg_proj,
    }
    checkpoint_metadata = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_type": "fcpg_corrector",
        "corrector_arch": args.corrector_arch,
        "policy_arch": None,
        "lewm_upstream_commit": LEWM_UPSTREAM_COMMIT,
        "has_reliability": args.corrector_arch == "legacy",
        "args": vars(args),
        "obs_dim": train_dataset.obs_dim,
        "num_keypoints": train_dataset.num_keypoints,
        "obs_mode": args.obs_mode,
        "data": str(Path(args.data).resolve()),
        "vision_cache": (
            str(Path(args.vision_cache).resolve()) if args.vision_cache else None
        ),
        "data_sources": [item["data"] for item in source_manifest],
        "vision_caches": [item["vision_cache"] for item in source_manifest],
        "source_manifest": source_manifest,
        "balanced_sources": bool(train_sampler is not None),
        "alignment": (
            alignments[0] if len(set(alignments)) == 1 else alignments
        ),
        "loss_config": loss_config,
    }
    history: list[dict[str, float | int]] = []
    best_metrics: dict[str, float | int] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            scaler,
            args.reliability_sigma_px,
            args.reliability_weight,
            sigreg,
            args.sigreg_weight,
            args.grad_clip,
            args.reliability_corruption_prob,
            args.cutout_radius_px,
            args.corrector_arch,
        )
        scheduler.step()
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            None,
            scaler,
            args.reliability_sigma_px,
            args.reliability_weight,
            sigreg,
            args.sigreg_weight,
            args.grad_clip,
            0.0,
            args.cutout_radius_px,
            args.corrector_arch,
        )
        row: dict[str, float | int] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        row["lr"] = float(optimizer.param_groups[0]["lr"])
        history.append(row)
        improved = best_metrics is None or float(row["val_e_after"]) < float(
            best_metrics["val_e_after"]
        )
        if improved:
            best_metrics = dict(row)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        message = (
            f"epoch {epoch:03d} "
            f"train_loss={row['train_loss']:.5f} "
            f"val_before={row['val_e_before']:.3f}px "
            f"val_after={row['val_e_after']:.3f}px "
            f"ratio={row['val_ratio']:.3f} "
        )
        if args.corrector_arch == "legacy":
            message += f"rho={row['val_reliability']:.3f} "
        message += f"lr={row['lr']:.2e}"
        print(message, flush=True)
        if improved and best_state is not None:
            atomic_torch_save(
                {**checkpoint_metadata, "model_state": best_state},
                best_path,
            )
        atomic_json_write(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "corrector_arch": args.corrector_arch,
                "lewm_upstream_commit": LEWM_UPSTREAM_COMMIT,
                "loss_config": loss_config,
                "best": best_metrics,
                "history": history,
                "complete": False,
            },
            metrics_path,
        )

    atomic_torch_save(
        {**checkpoint_metadata, "model_state": model.state_dict()},
        final_path,
    )
    atomic_json_write(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "corrector_arch": args.corrector_arch,
            "lewm_upstream_commit": LEWM_UPSTREAM_COMMIT,
            "loss_config": loss_config,
            "best": best_metrics,
            "history": history,
            "complete": True,
        },
        metrics_path,
    )
    print(f"final_checkpoint: {final_path}")
    print(f"best_checkpoint: {best_path}")
    print(f"metrics: {metrics_path}")
    if best_metrics is not None:
        print(
            f"best_epoch: {best_metrics['epoch']} "
            f"best_val_after: {best_metrics['val_e_after']:.4f}px "
            f"best_ratio: {best_metrics['val_ratio']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
