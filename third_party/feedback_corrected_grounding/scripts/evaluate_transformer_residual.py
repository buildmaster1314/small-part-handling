#!/usr/bin/env python3
"""Evaluate FCPG residual correction and its dependence on visual evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.geometric_baselines import refine_points_with_prototypes  # noqa: E402
from fcpg.transformer_grounding import DEFAULT_VIT_MODEL, load_fcpg_vit_adapter  # noqa: E402
from scripts.train_bc_policy import make_split  # noqa: E402
from scripts.train_transformer_residual import TransformerResidualDataset  # noqa: E402


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def validate_vision_cache(
    vision_cache: str | Path,
    data_path: str | Path,
    checkpoint_args: dict[str, object],
) -> None:
    metadata = json.loads(
        (Path(vision_cache) / "metadata.json").read_text(encoding="utf-8")
    )
    source_data = metadata.get("source_data")
    if source_data and Path(str(source_data)).resolve() != Path(data_path).resolve():
        raise ValueError("vision cache source does not match evaluation dataset")
    cached_model = str(metadata["model_name"])
    expected_model = str(checkpoint_args.get("model_name", DEFAULT_VIT_MODEL))
    if cached_model != expected_model:
        raise ValueError(
            f"vision cache model {cached_model!r} does not match checkpoint "
            f"model {expected_model!r}"
        )
    cached_layers = tuple(int(layer) for layer in metadata["selected_layers"])
    expected_layers = tuple(
        int(layer)
        for layer in checkpoint_args.get("vision_selected_layers", (2, 5, 8, 11))
    )
    if cached_layers != expected_layers:
        raise ValueError(
            f"vision cache layers {cached_layers} do not match checkpoint "
            f"layers {expected_layers}"
        )


def cutout_anchor_regions(
    rgb: torch.Tensor,
    q_nominal: torch.Tensor,
    radius_px: float,
) -> torch.Tensor:
    height, width = rgb.shape[-2:]
    ys = torch.arange(height, device=rgb.device, dtype=q_nominal.dtype)
    xs = torch.arange(width, device=rgb.device, dtype=q_nominal.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dx = (xx[None, None] - q_nominal[..., 0, None, None]).abs()
    dy = (yy[None, None] - q_nominal[..., 1, None, None]).abs()
    mask = ((dx <= radius_px) & (dy <= radius_px)).any(dim=1)
    return torch.where(mask[:, None], torch.zeros_like(rgb), rgb)


def evaluate_mode(
    model,
    loader: DataLoader,
    device: torch.device,
    mode: str,
    seed: int,
    cutout_radius_px: float,
) -> dict[str, float | bool | None]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    before_values: list[np.ndarray] = []
    after_values: list[np.ndarray] = []
    reliability_values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(device, non_blocking=device.type == "cuda")
                for key, value in batch.items()
            }
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                if "rgb" in batch:
                    rgb = batch["rgb"]
                    if mode == "zero-image":
                        rgb = torch.zeros_like(rgb)
                    elif mode == "shuffled-image":
                        permutation = torch.randperm(
                            rgb.shape[0], generator=generator
                        ).to(device)
                        rgb = rgb[permutation]
                    elif mode == "local-cutout":
                        rgb = cutout_anchor_regions(
                            rgb, batch["q_nominal"], cutout_radius_px
                        )
                    elif mode != "normal":
                        raise ValueError(f"unknown mode {mode!r}")
                    output = model(
                        rgb,
                        batch["q_nominal"],
                        batch["keypoints_xyz"],
                        batch["obs"],
                    )
                else:
                    if mode != "normal":
                        raise ValueError(
                            "cached feature evaluation currently supports only normal mode"
                        )
                    image_sizes = batch["image_size"]
                    if not bool(torch.all(image_sizes == image_sizes[0])):
                        raise ValueError(
                            "all cached residual samples must share one image size"
                        )
                    image_size = tuple(int(value) for value in image_sizes[0].tolist())
                    output = model.forward_from_backbone_features(
                        batch["vision_backbone_features"],
                        batch["vision_prefix"],
                        batch["q_nominal"],
                        batch["keypoints_xyz"],
                        batch["obs"],
                        image_size=image_size,
                    )
            valid = batch["valid"].detach().cpu().numpy().astype(bool)
            before = torch.linalg.vector_norm(
                batch["q_nominal"] - batch["q_true"], dim=-1
            ).detach().cpu().numpy()
            after = torch.linalg.vector_norm(
                output.q_corrected_px - batch["q_true"], dim=-1
            ).detach().cpu().numpy()
            reliability = getattr(output, "reliability", None)
            before_values.append(before[valid])
            after_values.append(after[valid])
            if reliability is not None:
                reliability_values.append(
                    reliability.detach().cpu().numpy()[valid]
                )
    before = np.concatenate(before_values)
    after = np.concatenate(after_values)
    reliability = (
        np.concatenate(reliability_values) if reliability_values else None
    )
    return {
        "e_before_mean_px": float(before.mean()),
        "e_after_mean_px": float(after.mean()),
        "ratio": float(after.mean() / max(before.mean(), 1e-8)),
        "e_after_median_px": float(np.median(after)),
        "e_after_p90_px": float(np.percentile(after, 90)),
        "has_reliability": reliability is not None,
        "reliability_mean": (
            float(reliability.mean()) if reliability is not None else None
        ),
    }


def load_prototype_bank(
    directory: str | Path,
    checkpoint_args: dict[str, object],
    image_size: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, object], str]:
    directory = Path(directory)
    descriptors_path = directory / "prototypes.npy"
    metadata_path = directory / "metadata.json"
    if not descriptors_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"invalid prototype bank directory: {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    bank_size = (int(metadata["image_width"]), int(metadata["image_height"]))
    if bank_size != image_size:
        raise ValueError(f"prototype image size {bank_size} does not match {image_size}")
    expected_model = str(checkpoint_args.get("model_name", DEFAULT_VIT_MODEL))
    if str(metadata["model_name"]) != expected_model:
        raise ValueError("prototype model does not match checkpoint")
    expected_layers = tuple(
        int(layer)
        for layer in checkpoint_args.get("vision_selected_layers", (2, 5, 8, 11))
    )
    bank_layers = tuple(int(layer) for layer in metadata["selected_layers"])
    if bank_layers != expected_layers:
        raise ValueError("prototype layers do not match checkpoint")
    descriptors = torch.from_numpy(np.load(descriptors_path)).to(device)
    digest = hashlib.sha256(descriptors_path.read_bytes()).hexdigest()
    return descriptors, metadata, digest


@torch.inference_mode()
def evaluate_prototype_baseline(
    loader: DataLoader,
    descriptors: torch.Tensor,
    feature_level: int,
    device: torch.device,
    radius_px: float,
    grid_size: int,
    temperature: float,
    min_confidence: float,
) -> dict[str, float]:
    before_values: list[np.ndarray] = []
    after_values: list[np.ndarray] = []
    confidence_values: list[np.ndarray] = []
    valid_match_values: list[np.ndarray] = []
    for batch in loader:
        if "vision_backbone_features" not in batch:
            raise ValueError("prototype baseline requires --vision-cache")
        batch = {
            key: value.to(device, non_blocking=device.type == "cuda")
            for key, value in batch.items()
        }
        image_sizes = batch["image_size"]
        if not bool(torch.all(image_sizes == image_sizes[0])):
            raise ValueError("prototype batch must share one image size")
        image_size = tuple(int(value) for value in image_sizes[0].tolist())
        feature_maps = batch["vision_backbone_features"]
        level = feature_level % feature_maps.shape[1]
        result = refine_points_with_prototypes(
            feature_maps[:, level],
            batch["q_nominal"],
            descriptors,
            image_size=image_size,
            radius_px=radius_px,
            grid_size=grid_size,
            temperature=temperature,
            min_confidence=min_confidence,
        )
        valid = batch["valid"].detach().cpu().numpy().astype(bool)
        before = torch.linalg.vector_norm(
            batch["q_nominal"] - batch["q_true"], dim=-1
        ).detach().cpu().numpy()
        after = torch.linalg.vector_norm(
            result.points_px - batch["q_true"], dim=-1
        ).detach().cpu().numpy()
        before_values.append(before[valid])
        after_values.append(after[valid])
        confidence_values.append(result.confidence.detach().cpu().numpy()[valid])
        valid_match_values.append(result.valid.detach().cpu().numpy()[valid])
    before = np.concatenate(before_values)
    after = np.concatenate(after_values)
    confidence = np.concatenate(confidence_values)
    match_valid = np.concatenate(valid_match_values)
    return {
        "e_before_mean_px": float(before.mean()),
        "e_after_mean_px": float(after.mean()),
        "ratio": float(after.mean() / max(before.mean(), 1e-8)),
        "e_after_median_px": float(np.median(after)),
        "e_after_p90_px": float(np.percentile(after, 90)),
        "match_confidence_mean": float(confidence.mean()),
        "match_valid_fraction": float(match_valid.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vision-cache", default="")
    parser.add_argument("--prototype-bank", default="")
    parser.add_argument("--prototype-radius-px", type=float, default=32.0)
    parser.add_argument("--prototype-grid-size", type=int, default=9)
    parser.add_argument("--prototype-temperature", type=float, default=0.0)
    parser.add_argument("--prototype-min-confidence", type=float, default=-1.0)
    parser.add_argument("--output", default="")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cutout-radius-px", type=float, default=12.0)
    parser.add_argument(
        "--modes",
        default="normal,local-cutout,zero-image,shuffled-image",
        help="Comma-separated evaluation modes. Cached features support normal only.",
    )
    args = parser.parse_args()

    modes = tuple(mode.strip() for mode in args.modes.split(",") if mode.strip())
    valid_modes = {"normal", "local-cutout", "zero-image", "shuffled-image"}
    if not modes or any(mode not in valid_modes for mode in modes):
        parser.error(f"--modes must contain values from {sorted(valid_modes)}")
    if args.vision_cache and not modes == ("normal",):
        parser.error("--vision-cache currently requires --modes normal")
    if args.prototype_bank and not args.vision_cache:
        parser.error("--prototype-bank requires --vision-cache")

    device = choose_device(args.device)
    model, checkpoint_args = load_fcpg_vit_adapter(args.checkpoint, device)
    if args.vision_cache:
        validate_vision_cache(args.vision_cache, args.data, checkpoint_args)
    obs_mode = str(checkpoint_args.get("obs_mode", "robot"))
    with np.load(args.data, allow_pickle=False) as raw:
        split = make_split(raw, args.val_fraction, args.split_seed, "task-episode")
    dataset = TransformerResidualDataset(
        args.data,
        split.test,
        obs_mode,
        vision_cache=args.vision_cache or None,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    results = {
        mode: evaluate_mode(model, loader, device, mode, args.seed, args.cutout_radius_px)
        for mode in modes
    }
    prototype_result = None
    prototype_metadata = None
    prototype_sha256 = None
    if args.prototype_bank:
        descriptors, prototype_metadata, prototype_sha256 = load_prototype_bank(
            args.prototype_bank,
            checkpoint_args,
            dataset.image_size,
            device,
        )
        prototype_result = evaluate_prototype_baseline(
            loader,
            descriptors,
            int(prototype_metadata["feature_level"]),
            device,
            args.prototype_radius_px,
            args.prototype_grid_size,
            args.prototype_temperature,
            args.prototype_min_confidence,
        )
    payload = {
        "data": args.data,
        "checkpoint": args.checkpoint,
        "vision_cache": args.vision_cache or None,
        "obs_mode": obs_mode,
        "validation_frames": len(dataset),
        "results": results,
        "prototype_baseline": prototype_result,
        "prototype_bank": str(Path(args.prototype_bank).resolve()) if args.prototype_bank else None,
        "prototype_bank_sha256": prototype_sha256,
        "prototype_settings": (
            {
                "radius_px": args.prototype_radius_px,
                "grid_size": args.prototype_grid_size,
                "temperature": args.prototype_temperature,
                "min_confidence": args.prototype_min_confidence,
            }
            if args.prototype_bank
            else None
        ),
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
