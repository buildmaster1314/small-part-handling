#!/usr/bin/env python3
"""Train the deterministic transformer action-chunk policy."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.action_chunk_policy import (  # noqa: E402
    GRIPPER_ACTION_MODES,
    GROUNDING_MODES,
    POLICY_ARCHES,
    RELIABILITY_OVERRIDES,
    GroundedActionChunkTransformer,
)
from fcpg.lewm_core import LEWM_UPSTREAM_COMMIT  # noqa: E402
from fcpg.observations import OBSERVATION_MODES  # noqa: E402
from fcpg.regularization import SIGReg  # noqa: E402
from fcpg.sequence_data import (  # noqa: E402
    HomogeneousConcatDataset,
    ProjectionSequenceDataset,
    compute_multi_source_normalization_stats,
    make_balanced_source_sampler,
    resolve_projection_sources,
)
from fcpg.transformer_grounding import load_fcpg_vit_adapter  # noqa: E402
from scripts.train_bc_policy import make_split, require_pre_action_alignment  # noqa: E402


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


def apply_projection_jitter(
    batch: dict[str, torch.Tensor],
    std_px: float,
    max_px: float,
) -> dict[str, torch.Tensor]:
    """Apply one coherent 2-D projection shift per sequence.

    Camera calibration errors move all projected keypoints coherently over the
    short observation horizon. Sharing the sampled shift across time and
    keypoints makes this a stronger, physically closer augmentation baseline
    than independent point noise.
    """

    if std_px <= 0:
        return batch
    q_nominal = batch["q_nominal"]
    if q_nominal.ndim != 4 or q_nominal.shape[-1] != 2:
        raise ValueError("q_nominal must have shape [B, H, K, 2]")
    shift = torch.randn(
        q_nominal.shape[0],
        1,
        1,
        2,
        device=q_nominal.device,
        dtype=q_nominal.dtype,
    ) * float(std_px)
    if max_px > 0:
        norm = torch.linalg.vector_norm(shift, dim=-1, keepdim=True)
        scale = (float(max_px) / norm.clamp_min(1e-12)).clamp(max=1.0)
        shift = shift * scale
    jittered = dict(batch)
    jittered["q_nominal"] = q_nominal + shift
    if "next_q_nominal" in batch:
        next_q_nominal = batch["next_q_nominal"]
        if next_q_nominal.shape != q_nominal.shape:
            raise ValueError("next_q_nominal must match q_nominal shape")
        jittered["next_q_nominal"] = next_q_nominal + shift
    return jittered


def checkpoint_state(model: GroundedActionChunkTransformer) -> dict[str, torch.Tensor]:
    frozen_prefixes = ["fcpg_adapter."]
    if model.vision_encoder.backbone_fully_frozen:
        frozen_prefixes.append("vision_encoder.backbone.")
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if not key.startswith(tuple(frozen_prefixes))
    }


def cosine_warmup(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(warmup_steps, 1))
    progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def compute_sigreg_loss(
    sigreg: SIGReg,
    representation: torch.Tensor,
    source_index: torch.Tensor | None,
    scope: str,
) -> torch.Tensor:
    if scope == "pooled":
        return sigreg(representation)
    if scope != "per-source":
        raise ValueError(f"unknown SIGReg scope: {scope}")
    if source_index is None:
        raise ValueError("per-source SIGReg requires source_index in every batch")
    losses = []
    for source in torch.unique(source_index, sorted=True):
        mask = source_index == source
        if int(mask.sum()) < 2:
            raise ValueError(
                "per-source SIGReg requires at least two samples from every source"
            )
        selected = (
            representation[mask]
            if representation.ndim == 2
            else representation[:, mask]
        )
        losses.append(sigreg(selected))
    return torch.stack(losses).mean()


def run_epoch(
    model: GroundedActionChunkTransformer,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    sigreg: SIGReg | None,
    sigreg_weight: float,
    sigreg_scope: str,
    gripper_weight: float,
    latent_pred_weight: float,
    grad_clip: float,
    max_batches: int,
    projection_jitter_std_px: float,
    projection_jitter_max_px: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "motion": 0.0, "gripper": 0.0, "sigreg": 0.0}
    if model.policy_arch == "lewm-predictive":
        totals["next_latent"] = 0.0
    count = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_index, batch in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            batch = move_batch(batch, device)
            if training:
                batch = apply_projection_jitter(
                    batch,
                    projection_jitter_std_px,
                    projection_jitter_max_px,
                )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                base_loss, parts, output = model.compute_loss(
                    batch,
                    gripper_weight=gripper_weight,
                    latent_pred_weight=latent_pred_weight,
                )
                if training and sigreg is not None and sigreg_weight > 0:
                    representation = (
                        output.state_latent
                        if output.state_latent is not None
                        else output.latent.transpose(0, 1).contiguous()
                    )
                    sigreg_loss = compute_sigreg_loss(
                        sigreg,
                        representation,
                        batch.get("source_index"),
                        sigreg_scope,
                    )
                else:
                    sigreg_loss = base_loss.new_zeros(())
                loss = base_loss + float(sigreg_weight) * sigreg_loss
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    max_norm=grad_clip,
                )
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_ran = scaler.get_scale() >= scale_before
                if scheduler is not None and optimizer_ran:
                    scheduler.step()
            batch_size = int(batch["obs"].shape[0])
            totals["loss"] += float(loss.detach()) * batch_size
            totals["motion"] += float(parts["motion_loss"].detach()) * batch_size
            totals["gripper"] += float(parts["gripper_loss"].detach()) * batch_size
            totals["sigreg"] += float(sigreg_loss.detach()) * batch_size
            if "next_latent_loss" in parts:
                totals["next_latent"] += (
                    float(parts["next_latent_loss"].detach()) * batch_size
                )
            count += batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_json_write(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def resolve_policy_weights(
    policy_arch: str,
    latent_pred_weight: float | None,
    sigreg_weight: float | None,
) -> tuple[float, float]:
    if policy_arch not in POLICY_ARCHES:
        raise ValueError(f"unknown policy architecture: {policy_arch}")
    if policy_arch == "legacy":
        resolved_latent = 0.0 if latent_pred_weight is None else float(latent_pred_weight)
        resolved_sigreg = 0.0 if sigreg_weight is None else float(sigreg_weight)
        if resolved_latent != 0.0:
            raise ValueError("legacy policy requires --latent-pred-weight 0")
    else:
        resolved_latent = 1.0 if latent_pred_weight is None else float(latent_pred_weight)
        resolved_sigreg = 0.02 if sigreg_weight is None else float(sigreg_weight)
        if not (
            math.isclose(resolved_sigreg, 0.0, abs_tol=1e-12)
            or math.isclose(resolved_sigreg, 0.02, rel_tol=1e-9, abs_tol=1e-12)
            or math.isclose(resolved_sigreg, 0.09, rel_tol=1e-9, abs_tol=1e-12)
        ):
            raise ValueError(
                "predictive policy supports matched SIGReg weights 0, 0.02, or 0.09"
            )
    if resolved_latent < 0 or resolved_sigreg < 0:
        raise ValueError("policy loss weights must be non-negative")
    return resolved_latent, resolved_sigreg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=GROUNDING_MODES, required=True)
    parser.add_argument("--policy-arch", choices=POLICY_ARCHES, default="legacy")
    parser.add_argument("--fcpg-checkpoint", default="")
    parser.add_argument("--vision-cache", default="")
    parser.add_argument("--grounding-cache", default="")
    parser.add_argument("--policy-cache", default="")
    parser.add_argument("--extra-data", action="append", default=[])
    parser.add_argument("--extra-vision-cache", action="append", default=[])
    parser.add_argument("--extra-grounding-cache", action="append", default=[])
    parser.add_argument("--extra-policy-cache", action="append", default=[])
    parser.add_argument(
        "--balance-sources",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reliability-override",
        choices=RELIABILITY_OVERRIDES,
        default="native",
    )
    parser.add_argument("--vision-model-name", default="vit_base_patch16_224.augreg_in1k")
    parser.add_argument("--vision-selected-layers", type=parse_layers, default=(2, 5, 8, 11))
    parser.add_argument("--obs-mode", choices=OBSERVATION_MODES, default="robot")
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--vision-dim", type=int, default=256)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=6)
    parser.add_argument("--predictor-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--trainable-backbone-blocks", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--gripper-weight", type=float, default=0.5)
    parser.add_argument(
        "--gripper-action-mode",
        choices=GRIPPER_ACTION_MODES,
        default="signed",
    )
    parser.add_argument("--latent-pred-weight", type=float, default=None)
    parser.add_argument("--sigreg-weight", type=float, default=None)
    parser.add_argument(
        "--sigreg-scope",
        choices=("pooled", "per-source"),
        default="pooled",
    )
    parser.add_argument("--sigreg-proj", type=int, default=256)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--projection-jitter-std-px", type=float, default=0.0)
    parser.add_argument("--projection-jitter-max-px", type=float, default=0.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    try:
        args.latent_pred_weight, args.sigreg_weight = resolve_policy_weights(
            args.policy_arch,
            args.latent_pred_weight,
            args.sigreg_weight,
        )
    except ValueError as error:
        parser.error(str(error))

    if args.mode == "fcpg" and not args.fcpg_checkpoint:
        parser.error("--fcpg-checkpoint is required for --mode fcpg")
    if args.projection_jitter_std_px < 0 or args.projection_jitter_max_px < 0:
        parser.error("projection jitter values must be non-negative")
    if args.projection_jitter_std_px > 0 and args.mode != "geoprop":
        parser.error("projection jitter augmentation is only valid for --mode geoprop")
    if args.projection_jitter_max_px > 0 and args.projection_jitter_std_px == 0:
        parser.error("--projection-jitter-max-px requires positive jitter std")
    try:
        sources = resolve_projection_sources(
            data=args.data,
            vision_cache=args.vision_cache or None,
            extra_data=args.extra_data,
            extra_vision_cache=args.extra_vision_cache,
            grounding_cache=args.grounding_cache or None,
            extra_grounding_cache=args.extra_grounding_cache,
            policy_cache=args.policy_cache or None,
            extra_policy_cache=args.extra_policy_cache,
        )
    except ValueError as error:
        parser.error(str(error))
    uses_policy_cache = any(source.policy_cache is not None for source in sources)
    if uses_policy_cache and not all(
        source.policy_cache is not None for source in sources
    ):
        parser.error("all sources must provide --policy-cache when any source uses it")
    if uses_policy_cache and any(source.vision_cache is not None for source in sources):
        parser.error("--policy-cache replaces --vision-cache during policy training")
    if uses_policy_cache and args.mode != "fcpg":
        parser.error("compact policy caches currently require --mode fcpg")
    if (
        any(source.vision_cache is not None for source in sources)
        and args.trainable_backbone_blocks > 0
    ):
        parser.error("--vision-cache requires a frozen ViT backbone")
    if (
        args.mode == "fcpg"
        and any(source.vision_cache is not None for source in sources)
        and any(source.grounding_cache is None for source in sources)
    ):
        parser.error("cached FCPG training requires one grounding cache per source")
    if args.mode != "fcpg" and any(
        source.grounding_cache is not None for source in sources
    ):
        parser.error("--grounding-cache is only valid for --mode fcpg")
    if args.reliability_override != "native" and args.mode != "fcpg":
        parser.error("--reliability-override is only valid for --mode fcpg")
    for source in sources:
        if source.vision_cache is not None:
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
            if str(cache_metadata["model_name"]) != args.vision_model_name:
                parser.error("vision cache model does not match --vision-model-name")
        if source.policy_cache is not None:
            policy_metadata = json.loads(
                (source.policy_cache / "metadata.json").read_text(encoding="utf-8")
            )
            if policy_metadata.get("cache_type") != "fcpg_policy_visual":
                parser.error("unsupported compact policy cache type")
            if policy_metadata.get("point_source") != "fcpg_q_corrected":
                parser.error("FCPG policy cache must sample fcpg_q_corrected points")
            if str(policy_metadata.get("model_name")) != args.vision_model_name:
                parser.error("policy cache model does not match --vision-model-name")
            source_data = policy_metadata.get("source_data")
            if source_data and Path(source_data).resolve() != source.data.resolve():
                parser.error(
                    f"policy cache source {source_data} does not match dataset "
                    f"{source.data.resolve()}"
                )
            if source.grounding_cache is None:
                parser.error("compact policy cache requires a grounding cache")
            cached_grounding = policy_metadata.get("grounding_cache")
            if (
                not cached_grounding
                or Path(cached_grounding).resolve() != source.grounding_cache.resolve()
            ):
                parser.error("policy cache grounding source does not match --grounding-cache")
            cached_checkpoint = policy_metadata.get("checkpoint")
            if (
                not cached_checkpoint
                or Path(cached_checkpoint).resolve()
                != Path(args.fcpg_checkpoint).resolve()
            ):
                parser.error("policy cache checkpoint does not match --fcpg-checkpoint")
            if int(policy_metadata["visual_dim"]) != args.vision_dim:
                parser.error("policy cache visual dimension does not match --vision-dim")
            cached_layers = tuple(
                int(layer) for layer in policy_metadata["selected_layers"]
            )
            if cached_layers != args.vision_selected_layers:
                parser.error("policy cache layers do not match --vision-selected-layers")
    args.reuse_fcpg_vision = bool(uses_policy_cache)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    source_splits = []
    alignments = []
    for source in sources:
        with np.load(source.data, allow_pickle=False) as raw:
            alignments.append(require_pre_action_alignment(raw, source.data, False))
            source_splits.append(
                make_split(raw, args.val_fraction, args.split_seed, "task-episode")
            )
    train_sources = [
        ProjectionSequenceDataset(
            source.data,
            split.train,
            obs_mode=args.obs_mode,
            obs_horizon=args.obs_horizon,
            action_horizon=args.action_horizon,
            vision_cache=source.vision_cache,
            grounding_cache=source.grounding_cache,
            policy_cache=source.policy_cache,
            source_index=source_index,
        )
        for source_index, (source, split) in enumerate(
            zip(sources, source_splits, strict=True)
        )
    ]
    val_sources = [
        ProjectionSequenceDataset(
            source.data,
            split.test,
            obs_mode=args.obs_mode,
            obs_horizon=args.obs_horizon,
            action_horizon=args.action_horizon,
            vision_cache=source.vision_cache,
            grounding_cache=source.grounding_cache,
            policy_cache=source.policy_cache,
            source_index=source_index,
        )
        for source_index, (source, split) in enumerate(
            zip(sources, source_splits, strict=True)
        )
    ]
    if len(train_sources) == 1:
        train_dataset = train_sources[0]
        val_dataset = val_sources[0]
    else:
        shared_attributes = (
            "obs_dim",
            "action_dim",
            "num_keypoints",
            "image_size",
            "policy_visual_dim",
        )
        train_dataset = HomogeneousConcatDataset(train_sources, shared_attributes)
        val_dataset = HomogeneousConcatDataset(val_sources, shared_attributes)
    stats = compute_multi_source_normalization_stats(
        [source.data for source in sources],
        [split.train for split in source_splits],
        args.obs_mode,
    )
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
    fcpg_adapter = None
    if args.mode == "fcpg":
        fcpg_adapter, fcpg_args = load_fcpg_vit_adapter(args.fcpg_checkpoint, device)
        corrector_obs_mode = str(fcpg_args.get("obs_mode", "robot"))
        compatible_obs_modes = (
            corrector_obs_mode == args.obs_mode
            or (corrector_obs_mode == "robot" and args.obs_mode == "robot_goal")
        )
        if not compatible_obs_modes:
            raise ValueError(
                "FCPG adapter and action policy observation modes are incompatible: "
                f"corrector={corrector_obs_mode!r}, policy={args.obs_mode!r}"
            )
    model = GroundedActionChunkTransformer(
        mode=args.mode,
        obs_dim=train_dataset.obs_dim,
        action_dim=train_dataset.action_dim,
        num_keypoints=train_dataset.num_keypoints,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        model_dim=args.model_dim,
        vision_dim=args.vision_dim,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        vision_model_name=args.vision_model_name,
        vision_selected_layers=args.vision_selected_layers,
        trainable_backbone_blocks=args.trainable_backbone_blocks,
        fcpg_adapter=fcpg_adapter,
        policy_arch=args.policy_arch,
        predictor_layers=args.predictor_layers,
        reuse_fcpg_vision=args.reuse_fcpg_vision,
        gripper_action_mode=args.gripper_action_mode,
    ).to(device)
    model.set_reliability_override(args.reliability_override)
    model.set_normalization(stats)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    backbone_parameters = [
        parameter for parameter in model.vision_encoder.backbone.parameters() if parameter.requires_grad
    ]
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    head_parameters = [parameter for parameter in trainable if id(parameter) not in backbone_ids]
    parameter_groups = [{"params": head_parameters, "lr": args.lr}]
    if backbone_parameters:
        parameter_groups.append({"params": backbone_parameters, "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=args.lr,
        betas=(0.95, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    train_batches = len(train_loader) if args.max_train_batches <= 0 else min(len(train_loader), args.max_train_batches)
    total_steps = max(1, args.epochs * train_batches)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup(step, args.warmup_steps, total_steps),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    sigreg = SIGReg(num_proj=args.sigreg_proj).to(device) if args.sigreg_weight > 0 else None

    source_manifest = [
        {
            "data": str(source.data.resolve()),
            "vision_cache": (
                str(source.vision_cache.resolve())
                if source.vision_cache is not None
                else None
            ),
            "grounding_cache": (
                str(source.grounding_cache.resolve())
                if source.grounding_cache is not None
                else None
            ),
            "policy_cache": (
                str(source.policy_cache.resolve())
                if source.policy_cache is not None
                else None
            ),
            "alignment": alignment,
            "train_sequences": int(len(train_source)),
            "val_sequences": int(len(val_source)),
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
    print(
        f"mode: {args.mode} policy_arch: {args.policy_arch} "
        f"obs_mode: {args.obs_mode} device: {device}"
    )
    print(
        f"loss_weights: latent_pred={args.latent_pred_weight} "
        f"sigreg={args.sigreg_weight} sigreg_scope={args.sigreg_scope} "
        f"gripper={args.gripper_weight} action_mode={args.gripper_action_mode}"
    )
    print(f"obs_horizon: {args.obs_horizon} action_horizon: {args.action_horizon}")
    print(
        f"vision_layers: {args.vision_selected_layers} "
        f"caches: {[item['vision_cache'] for item in source_manifest]}"
    )
    print(
        "grounding_caches: "
        f"{[item['grounding_cache'] for item in source_manifest]}"
    )
    print(
        "policy_caches: "
        f"{[item['policy_cache'] for item in source_manifest]}"
    )
    print(f"reliability_override: {args.reliability_override}")
    print(
        "projection_jitter_px: "
        f"std={args.projection_jitter_std_px} max={args.projection_jitter_max_px}"
    )
    print(f"batch_size: {args.batch_size} train: {len(train_dataset)} val: {len(val_dataset)}")
    print(f"parameters_total: {sum(parameter.numel() for parameter in model.parameters())}")
    print(f"parameters_trainable: {sum(parameter.numel() for parameter in trainable)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"{args.mode}_action_chunk.pt"
    best_path = output_dir / f"{args.mode}_action_chunk_best.pt"
    metrics_path = output_dir / f"{args.mode}_metrics.json"
    loss_config = {
        "motion": "smooth_l1_beta_0.05",
        "gripper_weight": args.gripper_weight,
        "latent_pred_weight": args.latent_pred_weight,
        "sigreg_weight": args.sigreg_weight,
        "sigreg_scope": args.sigreg_scope,
        "sigreg_proj": args.sigreg_proj,
    }
    checkpoint_metadata = {
        "schema_version": 2,
        "checkpoint_type": "fcpg_action_policy",
        "policy_arch": args.policy_arch,
        "lewm_upstream_commit": LEWM_UPSTREAM_COMMIT,
        "args": vars(args),
        "obs_dim": train_dataset.obs_dim,
        "action_dim": train_dataset.action_dim,
        "num_keypoints": train_dataset.num_keypoints,
        "normalization": stats.as_torch(),
        "loss_config": loss_config,
        "alignment": (
            alignments[0] if len(set(alignments)) == 1 else alignments
        ),
        "data": str(Path(args.data).resolve()),
        "vision_cache": (
            str(Path(args.vision_cache).resolve()) if args.vision_cache else None
        ),
        "grounding_cache": (
            str(Path(args.grounding_cache).resolve()) if args.grounding_cache else None
        ),
        "policy_cache": (
            str(Path(args.policy_cache).resolve()) if args.policy_cache else None
        ),
        "data_sources": [item["data"] for item in source_manifest],
        "vision_caches": [item["vision_cache"] for item in source_manifest],
        "grounding_caches": [item["grounding_cache"] for item in source_manifest],
        "policy_caches": [item["policy_cache"] for item in source_manifest],
        "source_manifest": source_manifest,
        "balanced_sources": bool(train_sampler is not None),
    }
    history = []
    best = None
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_started = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            scaler,
            scheduler,
            sigreg,
            args.sigreg_weight,
            args.sigreg_scope,
            args.gripper_weight,
            args.latent_pred_weight,
            args.grad_clip,
            args.max_train_batches,
            args.projection_jitter_std_px,
            args.projection_jitter_max_px,
        )
        train_seconds = time.perf_counter() - train_started
        val_started = time.perf_counter()
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            None,
            scaler,
            None,
            None,
            0.0,
            args.sigreg_scope,
            args.gripper_weight,
            args.latent_pred_weight,
            args.grad_clip,
            args.max_val_batches,
            0.0,
            0.0,
        )
        val_seconds = time.perf_counter() - val_started
        row = {"epoch": epoch, "lr": float(optimizer.param_groups[0]["lr"])}
        if len(optimizer.param_groups) > 1:
            row["backbone_lr"] = float(optimizer.param_groups[1]["lr"])
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        row["train_seconds"] = train_seconds
        row["val_seconds"] = val_seconds
        row["train_samples_per_second"] = (
            train_batches * args.batch_size / max(train_seconds, 1e-8)
        )
        if device.type == "cuda":
            row["peak_cuda_gb"] = torch.cuda.max_memory_allocated(device) / (1024**3)
        history.append(row)
        if best is None or row["val_loss"] < best["val_loss"]:
            best = dict(row)
            best_state = checkpoint_state(model)
            epochs_without_improvement = 0
            atomic_torch_save(
                {**checkpoint_metadata, "model_state": best_state},
                best_path,
            )
        else:
            epochs_without_improvement += 1
        print(
            f"epoch {epoch:03d} train={row['train_loss']:.5f} "
            f"val={row['val_loss']:.5f} motion={row['val_motion']:.5f} "
            f"gripper={row['val_gripper']:.5f} lr={row['lr']:.2e} "
            f"train_s={train_seconds:.2f} val_s={val_seconds:.2f} "
            f"samples_s={row['train_samples_per_second']:.0f} "
            f"peak_gb={row.get('peak_cuda_gb', 0.0):.2f}",
            flush=True,
        )
        atomic_json_write(
            {
                "schema_version": 2,
                "policy_arch": args.policy_arch,
                "lewm_upstream_commit": LEWM_UPSTREAM_COMMIT,
                "loss_config": loss_config,
                "best": best,
                "history": history,
                "complete": False,
            },
            metrics_path,
        )
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(
                f"early_stop: no validation improvement for {args.patience} epochs",
                flush=True,
            )
            break

    atomic_torch_save(
        {**checkpoint_metadata, "model_state": checkpoint_state(model)},
        final_path,
    )
    atomic_json_write(
        {
            "schema_version": 2,
            "policy_arch": args.policy_arch,
            "lewm_upstream_commit": LEWM_UPSTREAM_COMMIT,
            "loss_config": loss_config,
            "best": best,
            "history": history,
            "complete": True,
        },
        metrics_path,
    )
    print(f"final_checkpoint: {final_path}")
    print(f"best_checkpoint: {best_path}")
    print(f"metrics: {metrics_path}")
    if best is not None:
        print(f"best_epoch: {best['epoch']} best_val_loss: {best['val_loss']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
