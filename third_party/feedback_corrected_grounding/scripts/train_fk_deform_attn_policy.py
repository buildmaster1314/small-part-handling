#!/usr/bin/env python3
"""Train the matched policy-only residual and FK-DeformAttn pair.

The two variants are advanced on the same batches with identical optimizer,
scheduler, step, and checkpoint-selection contracts.  Projection targets are
discarded before a batch reaches either model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.fk_deform_attn_baseline import (  # noqa: E402
    AUDITED_VISION_MODEL,
    AUDITED_VISION_SELECTED_LAYERS,
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_TYPE,
    DEVELOPMENT_CLAIM_SCOPE,
    DEPLOYMENT_INPUT_KEYS,
    LOSS_CONTRACT,
    SELECTION_POLICY,
    STRUCTURAL_ATTRIBUTION_CONTRACT,
    VARIANTS,
    MatchedPolicyOnlyGrounding,
    benchmark_interleaved_latency,
    build_matched_policy_only_grounding,
    checkpoint_model_state,
    sha256_file,
    validate_structural_pair,
)
from fcpg.sequence_data import (  # noqa: E402
    HomogeneousConcatDataset,
    ProjectionSequenceDataset,
    compute_multi_source_normalization_stats,
    make_balanced_source_sampler,
    resolve_projection_sources,
)
from scripts.train_bc_policy import make_split, require_pre_action_alignment  # noqa: E402


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def cosine_warmup(step: int, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_json_write(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _state_sha256(
    module: torch.nn.Module,
    *,
    excluded_prefixes: tuple[str, ...] = (),
) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if name.startswith(excluded_prefixes):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_vision_cache_metadata(
    data_path: str | Path,
    cache_path: str | Path,
    *,
    expected_model_name: str = AUDITED_VISION_MODEL,
    expected_selected_layers: tuple[int, ...] = AUDITED_VISION_SELECTED_LAYERS,
) -> dict[str, object]:
    """Validate one audited cache against its source before dataset creation."""

    expected_layers = tuple(int(layer) for layer in expected_selected_layers)
    if expected_model_name != AUDITED_VISION_MODEL:
        raise ValueError(
            f"audited baseline requires ViT-B model {AUDITED_VISION_MODEL!r}"
        )
    if expected_layers != AUDITED_VISION_SELECTED_LAYERS:
        raise ValueError("audited baseline requires only ViT-B layer 11")
    source = Path(data_path).expanduser().resolve()
    cache = Path(cache_path).expanduser().resolve()
    metadata_path = cache / "metadata.json"
    if not source.is_file():
        raise FileNotFoundError(source)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError(f"vision cache metadata must be an object: {metadata_path}")

    cached_model = str(metadata.get("model_name", ""))
    if cached_model != expected_model_name:
        raise ValueError(
            f"vision cache model {cached_model!r} does not match declared "
            f"{expected_model_name!r}: {cache}"
        )
    try:
        cached_layers = tuple(int(layer) for layer in metadata["selected_layers"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"vision cache has invalid selected_layers: {cache}") from error
    if cached_layers != expected_layers:
        raise ValueError(
            f"vision cache layers {cached_layers} do not match declared "
            f"{expected_layers}: {cache}"
        )
    cached_source = metadata.get("source_data")
    if not isinstance(cached_source, str) or not cached_source:
        raise ValueError(f"vision cache must declare source_data: {cache}")
    if Path(cached_source).expanduser().resolve() != source:
        raise ValueError(
            f"vision cache source {cached_source} does not match declared {source}"
        )

    with np.load(source, allow_pickle=False) as data:
        if "obs" not in data.files or data["obs"].ndim == 0:
            raise ValueError(f"vision cache source has no sample-aligned obs: {source}")
        source_sample_count = int(data["obs"].shape[0])
    try:
        metadata_sample_count = int(metadata["sample_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"vision cache has invalid sample_count: {cache}") from error
    if metadata_sample_count != source_sample_count:
        raise ValueError(
            f"vision cache metadata sample_count {metadata_sample_count} does not match "
            f"source sample count {source_sample_count}: {cache}"
        )

    feature_path = cache / "backbone_features.npy"
    prefix_path = cache / "prefix.npy"
    if not feature_path.is_file() or not prefix_path.is_file():
        raise FileNotFoundError(f"vision cache arrays are incomplete: {cache}")
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    prefix = np.load(prefix_path, mmap_mode="r", allow_pickle=False)
    if int(features.shape[0]) != metadata_sample_count:
        raise ValueError("vision cache feature count does not match metadata sample_count")
    if int(prefix.shape[0]) != metadata_sample_count:
        raise ValueError("vision cache prefix count does not match metadata sample_count")
    try:
        declared_feature_shape = tuple(int(size) for size in metadata["feature_shape"])
        declared_prefix_dim = int(metadata["prefix_dim"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"vision cache has invalid feature shape metadata: {cache}") from error
    if tuple(features.shape[1:]) != declared_feature_shape:
        raise ValueError("vision cache feature array shape does not match metadata")
    if prefix.ndim != 2 or int(prefix.shape[1]) != declared_prefix_dim:
        raise ValueError("vision cache prefix array shape does not match metadata")

    return {
        "cache": str(cache),
        "metadata": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "source_data": str(source),
        "model_name": cached_model,
        "selected_layers": cached_layers,
        "sample_count": metadata_sample_count,
        "feature_shape": tuple(int(size) for size in features.shape[1:]),
        "prefix_dim": int(prefix.shape[1]),
    }


def policy_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move only deployment inputs and BC targets to the training device."""

    allowed = (*DEPLOYMENT_INPUT_KEYS, "action", "action_mask")
    selected = {
        key: batch[key].to(device, non_blocking=device.type == "cuda")
        for key in allowed
        if key in batch
    }
    for required in ("obs", "keypoints_xyz", "q_nominal", "action", "action_mask"):
        if required not in selected:
            raise KeyError(f"policy-only batch is missing {required!r}")
    return selected


def _rng_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    cpu = torch.random.get_rng_state()
    cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return cpu, cuda


def _set_rng_state(
    state: tuple[torch.Tensor, torch.Tensor | None],
    device: torch.device,
) -> None:
    torch.random.set_rng_state(state[0])
    if device.type == "cuda":
        assert state[1] is not None
        torch.cuda.set_rng_state(state[1], device)


def run_paired_epoch(
    models: dict[str, MatchedPolicyOnlyGrounding],
    loader: DataLoader,
    device: torch.device,
    *,
    optimizers: dict[str, torch.optim.Optimizer] | None,
    schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler] | None,
    gripper_weight: float,
    grad_clip: float,
    max_batches: int,
    amp: bool,
    microbatch_size: int = 0,
) -> tuple[dict[str, dict[str, float]], int]:
    training = optimizers is not None
    for model in models.values():
        model.train(training)
    totals = {
        variant: {"loss": 0.0, "motion": 0.0, "gripper": 0.0}
        for variant in VARIANTS
    }
    sample_count = 0
    batch_count = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_index, raw_batch in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            if "obs" not in raw_batch:
                raise KeyError("policy-only raw batch is missing 'obs'")
            batch_size = int(raw_batch["obs"].shape[0])
            chunk_size = batch_size if microbatch_size <= 0 else microbatch_size
            if chunk_size <= 0:
                raise ValueError("microbatch_size must be non-negative")
            action_mask = raw_batch.get("action_mask")
            if not isinstance(action_mask, torch.Tensor):
                raise KeyError("policy-only raw batch is missing 'action_mask'")
            total_valid = float(action_mask.float().sum())
            if total_valid <= 0.0:
                raise ValueError("policy-only batch has no valid action targets")
            batch_metrics = {
                variant: {"loss": 0.0, "motion": 0.0, "gripper": 0.0}
                for variant in VARIANTS
            }
            for variant in VARIANTS:
                optimizer = None if optimizers is None else optimizers[variant]
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)

            for start in range(0, batch_size, chunk_size):
                stop = min(start + chunk_size, batch_size)
                micro_raw = {
                    name: value[start:stop]
                    if isinstance(value, torch.Tensor)
                    and value.ndim > 0
                    and int(value.shape[0]) == batch_size
                    else value
                    for name, value in raw_batch.items()
                }
                batch = policy_batch(micro_raw, device)
                micro_valid = float(batch["action_mask"].float().sum())
                loss_weight = micro_valid / total_valid
                shared_rng = _rng_state(device)
                for variant in VARIANTS:
                    _set_rng_state(shared_rng, device)
                    model = models[variant]
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=amp and device.type == "cuda",
                    ):
                        loss, parts, _ = model.compute_bc_loss(
                            batch,
                            gripper_weight=gripper_weight,
                        )
                    if training:
                        (loss * loss_weight).backward()
                    batch_metrics[variant]["loss"] += (
                        float(loss.detach()) * loss_weight
                    )
                    batch_metrics[variant]["motion"] += (
                        float(parts["motion_loss"].detach()) * loss_weight
                    )
                    batch_metrics[variant]["gripper"] += (
                        float(parts["gripper_loss"].detach()) * loss_weight
                    )
                del batch

            for variant in VARIANTS:
                model = models[variant]
                optimizer = None if optimizers is None else optimizers[variant]
                if optimizer is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        ],
                        max_norm=grad_clip,
                    )
                    optimizer.step()
                    if schedulers is not None:
                        schedulers[variant].step()
                for name in ("loss", "motion", "gripper"):
                    totals[variant][name] += batch_metrics[variant][name] * batch_size
            sample_count += batch_size
            batch_count += 1
    metrics = {
        variant: {
            key: value / max(sample_count, 1)
            for key, value in totals[variant].items()
        }
        for variant in VARIANTS
    }
    return metrics, batch_count


def _single_item(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    cpu_single: dict[str, torch.Tensor] = {}
    for key in DEPLOYMENT_INPUT_KEYS:
        if key not in batch:
            continue
        value = batch[key]
        if value.device.type != "cpu":
            raise ValueError("_single_item requires a CPU DataLoader batch")
        cpu_single[key] = value[:1].clone()
    for required in ("obs", "keypoints_xyz", "q_nominal"):
        if required not in cpu_single:
            raise KeyError(f"latency batch is missing {required!r}")
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in cpu_single.items()
    }


def _load_active_state(
    model: MatchedPolicyOnlyGrounding,
    state: dict[str, torch.Tensor],
) -> None:
    incompatible = model.load_state_dict(state, strict=False)
    invalid_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("policy.vision_encoder.backbone.")
    ]
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"invalid active state: missing={invalid_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--data", default="")
    parser.add_argument("--vision-cache", default="")
    parser.add_argument("--extra-data", action="append", default=[])
    parser.add_argument("--extra-vision-cache", action="append", default=[])
    parser.add_argument("--vision-model-name", default=AUDITED_VISION_MODEL)
    parser.add_argument(
        "--vision-selected-layers",
        type=int,
        nargs="+",
        default=list(AUDITED_VISION_SELECTED_LAYERS),
    )
    parser.add_argument("--policy-contract-checkpoint", default="")
    parser.add_argument("--fcpg-corrector-checkpoint", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--obs-mode", default="robot")
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--lattice-step-px", type=float, default=1.0)
    parser.add_argument("--offset-scale-px", type=float, default=20.0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--microbatch-size",
        type=int,
        default=0,
        help="GPU microbatch; 0 keeps the full effective batch in one forward pass",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--gripper-weight", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evidence-scope", default=DEVELOPMENT_CLAIM_SCOPE)
    parser.add_argument(
        "--cross-seed-generalization-claim-supported",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--balance-sources", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--latency-iterations", type=int, default=50)
    parser.add_argument("--device", default="auto")
    return parser


def parse_args() -> argparse.Namespace:
    parser = make_parser()
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        config_path = Path(preliminary.config)
        values = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            parser.error("--config must contain a JSON object")
        valid = {action.dest for action in parser._actions}  # noqa: SLF001
        unknown = sorted(set(values) - valid)
        if unknown:
            parser.error(f"unknown config keys: {unknown}")
        parser.set_defaults(**values)
    args = parser.parse_args()
    missing = [
        name
        for name in (
            "data",
            "vision_cache",
            "policy_contract_checkpoint",
            "fcpg_corrector_checkpoint",
            "output_dir",
        )
        if not getattr(args, name)
    ]
    if missing:
        parser.error(f"missing required settings: {missing}")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be in (0,1)")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive")
    if args.microbatch_size < 0 or args.microbatch_size > args.batch_size:
        parser.error("--microbatch-size must be in [0, batch-size]")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.prefetch_factor <= 0:
        parser.error("--prefetch-factor must be positive")
    if args.lr <= 0 or args.weight_decay < 0:
        parser.error("optimizer settings must be non-negative with positive lr")
    if args.latency_warmup <= 0 or args.latency_iterations <= 0:
        parser.error("latency warmup and iterations must be positive")
    args.vision_selected_layers = tuple(int(layer) for layer in args.vision_selected_layers)
    if args.vision_model_name != AUDITED_VISION_MODEL:
        parser.error(f"audited baseline requires --vision-model-name {AUDITED_VISION_MODEL}")
    if args.vision_selected_layers != AUDITED_VISION_SELECTED_LAYERS:
        parser.error("audited baseline requires --vision-selected-layers 11")
    if args.evidence_scope != DEVELOPMENT_CLAIM_SCOPE or args.seed != 0:
        parser.error(
            "this audited config is limited to the seed0 development controlled comparison"
        )
    if args.cross_seed_generalization_claim_supported:
        parser.error("seed0 does not support a cross-seed generalization claim")
    return args


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    sources = resolve_projection_sources(
        args.data,
        args.vision_cache or None,
        extra_data=args.extra_data,
        extra_vision_cache=args.extra_vision_cache,
    )
    cache_records = []
    for source in sources:
        if source.vision_cache is None:
            raise ValueError("audited FK-DeformAttn training requires a cache per source")
        cache_records.append(
            validate_vision_cache_metadata(
                source.data,
                source.vision_cache,
                expected_model_name=args.vision_model_name,
                expected_selected_layers=args.vision_selected_layers,
            )
        )
    vision_cache_validation = {
        "status": "validated_before_training",
        "required_model_name": args.vision_model_name,
        "required_selected_layers": args.vision_selected_layers,
        "source_count": len(cache_records),
        "sources": cache_records,
    }
    claim_contract = {
        "scope": args.evidence_scope,
        "training_seed": int(args.seed),
        "development_only": True,
        "controlled_comparison_only": True,
        "cross_seed_generalization_supported": False,
        "general_claim_supported": False,
        "statement": (
            "seed0 is a development controlled comparison only; no cross-seed "
            "general claim is supported"
        ),
    }
    splits = []
    alignments = []
    for source in sources:
        with np.load(source.data, allow_pickle=False) as raw:
            alignments.append(require_pre_action_alignment(raw, source.data, False))
            splits.append(make_split(raw, args.val_fraction, args.split_seed, "task-episode"))
    train_sources = [
        ProjectionSequenceDataset(
            source.data,
            split.train,
            obs_mode=args.obs_mode,
            obs_horizon=args.obs_horizon,
            action_horizon=args.action_horizon,
            vision_cache=source.vision_cache,
            source_index=source_index,
        )
        for source_index, (source, split) in enumerate(zip(sources, splits, strict=True))
    ]
    val_sources = [
        ProjectionSequenceDataset(
            source.data,
            split.test,
            obs_mode=args.obs_mode,
            obs_horizon=args.obs_horizon,
            action_horizon=args.action_horizon,
            vision_cache=source.vision_cache,
            source_index=source_index,
        )
        for source_index, (source, split) in enumerate(zip(sources, splits, strict=True))
    ]
    if len(train_sources) == 1:
        train_dataset = train_sources[0]
        val_dataset = val_sources[0]
    else:
        shared = ("obs_dim", "action_dim", "num_keypoints", "image_size")
        train_dataset = HomogeneousConcatDataset(train_sources, shared)
        val_dataset = HomogeneousConcatDataset(val_sources, shared)
    stats = compute_multi_source_normalization_stats(
        [source.data for source in sources],
        [split.train for split in splits],
        args.obs_mode,
    )
    sampler = (
        make_balanced_source_sampler(train_sources, args.seed)
        if len(train_sources) > 1 and args.balance_sources
        else None
    )
    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_options["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options,
    )

    models: dict[str, MatchedPolicyOnlyGrounding] = {}
    build_contracts: dict[str, dict[str, object]] = {}
    for variant in VARIANTS:
        model, build_contract = build_matched_policy_only_grounding(
            variant=variant,
            policy_contract_checkpoint=args.policy_contract_checkpoint,
            fcpg_corrector_checkpoint=args.fcpg_corrector_checkpoint,
            policy_seed=args.seed,
            num_samples=args.num_samples,
            lattice_step_px=args.lattice_step_px,
            offset_scale_px=args.offset_scale_px,
            device=device,
        )
        config = build_contract["policy_config"]
        assert isinstance(config, dict)
        if str(config["vision_model_name"]) != args.vision_model_name:
            raise ValueError("policy contract ViT model does not match cache declaration")
        if tuple(config["vision_selected_layers"]) != args.vision_selected_layers:
            raise ValueError("policy contract ViT layers do not match cache declaration")
        expected = (
            str(config["obs_mode"]),
            int(config["obs_horizon"]),
            int(config["action_horizon"]),
            int(config["obs_dim"]),
            int(config["action_dim"]),
            int(config["num_keypoints"]),
        )
        observed = (
            args.obs_mode,
            args.obs_horizon,
            args.action_horizon,
            int(train_dataset.obs_dim),
            int(train_dataset.action_dim),
            int(train_dataset.num_keypoints),
        )
        if observed != expected:
            raise ValueError(
                f"dataset/policy contract mismatch: observed={observed}, expected={expected}"
            )
        model.set_normalization(stats.as_torch())
        models[variant] = model
        build_contracts[variant] = build_contract

    residual_model = models["policy_only_residual"]
    deform_model = models["fk_deform_attn"]
    deform_model.policy.vision_encoder.backbone = (
        residual_model.policy.vision_encoder.backbone
    )
    residual_policy_hash = _state_sha256(
        residual_model.policy,
        excluded_prefixes=("vision_encoder.backbone.",),
    )
    deform_policy_hash = _state_sha256(
        deform_model.policy,
        excluded_prefixes=("vision_encoder.backbone.",),
    )
    if residual_policy_hash != deform_policy_hash:
        raise RuntimeError("paired policies do not have identical initial state")
    counts = {variant: model.parameter_counts() for variant, model in models.items()}
    residual_counts = counts["policy_only_residual"]
    deform_counts = counts["fk_deform_attn"]
    for field in ("common_policy_active", "vision_neck_active"):
        if residual_counts[field] != deform_counts[field]:
            raise RuntimeError(f"paired parameter mismatch in {field}: {counts}")
    pair_active_gap = abs(
        residual_counts["total_active"] - deform_counts["total_active"]
    ) / float(max(residual_counts["total_active"], deform_counts["total_active"]))
    if pair_active_gap > 0.01:
        raise RuntimeError(f"paired active parameter gap exceeds 1%: {counts}")

    optimizers = {
        variant: torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.lr,
            betas=(0.95, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
        )
        for variant, model in models.items()
    }
    train_batches = len(train_loader)
    if args.max_train_batches > 0:
        train_batches = min(train_batches, args.max_train_batches)
    total_steps = max(1, args.epochs * train_batches)
    schedulers = {
        variant: torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: cosine_warmup(step, args.warmup_steps, total_steps),
        )
        for variant, optimizer in optimizers.items()
    }

    source_manifest = [
        {
            "data": str(source.data.resolve()),
            "vision_cache": (
                str(source.vision_cache.resolve()) if source.vision_cache else None
            ),
            "alignment": alignment,
            "train_sequences": int(len(train_source)),
            "val_sequences": int(len(val_source)),
            "vision_cache_validation": cache_record,
        }
        for source, alignment, train_source, val_source, cache_record in zip(
            sources,
            alignments,
            train_sources,
            val_sources,
            cache_records,
            strict=True,
        )
    ]
    data_contract = {
        "sources": source_manifest,
        "split_unit": "task_episode",
        "split_seed": int(args.split_seed),
        "val_fraction": float(args.val_fraction),
        "balanced_sources": bool(sampler is not None),
        "obs_mode": args.obs_mode,
        "obs_horizon": int(args.obs_horizon),
        "action_horizon": int(args.action_horizon),
        "model_batch_fields": (*DEPLOYMENT_INPUT_KEYS, "action", "action_mask"),
        "projection_targets_forwarded": False,
    }
    maximum_budget = {
        "epochs": int(args.epochs),
        "optimizer_steps_per_variant": int(total_steps),
    }
    selection_contract = {
        "policy": SELECTION_POLICY,
        "metric": "validation_behavior_cloning_loss",
        "direction": "minimize",
        "cadence": "every_epoch",
        "validation_split": "shared_complete_task_episodes",
        "pair_stopping_rule": "stop_only_after_both_variants_exhaust_patience",
        "maximum_budget": maximum_budget,
        "selected_step_relationship": "independent_by_variant",
        "same_selected_step_required": False,
    }
    pair_run_id = uuid.uuid4().hex
    print(f"device: {device}")
    print(f"sources: {json.dumps(source_manifest)}")
    print(f"vision_cache_validation: {json.dumps(vision_cache_validation)}")
    print(f"claim_contract: {json.dumps(claim_contract)}")
    print(f"initial_common_policy_sha256: {residual_policy_hash}")
    print(f"parameter_counts: {json.dumps(counts)}")
    print(f"loss_contract: {json.dumps(LOSS_CONTRACT)}")

    history: list[dict[str, object]] = []
    best_metrics: dict[str, dict[str, float | int]] = {}
    best_states: dict[str, dict[str, torch.Tensor]] = {}
    epochs_without_improvement = {variant: 0 for variant in VARIANTS}
    optimizer_steps = 0
    completed_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_started = time.perf_counter()
        train_metrics, steps = run_paired_epoch(
            models,
            train_loader,
            device,
            optimizers=optimizers,
            schedulers=schedulers,
            gripper_weight=args.gripper_weight,
            grad_clip=args.grad_clip,
            max_batches=args.max_train_batches,
            amp=args.amp,
            microbatch_size=args.microbatch_size,
        )
        train_seconds = time.perf_counter() - train_started
        optimizer_steps += steps
        val_started = time.perf_counter()
        val_metrics, _ = run_paired_epoch(
            models,
            val_loader,
            device,
            optimizers=None,
            schedulers=None,
            gripper_weight=args.gripper_weight,
            grad_clip=args.grad_clip,
            max_batches=args.max_val_batches,
            amp=args.amp,
            microbatch_size=args.microbatch_size,
        )
        val_seconds = time.perf_counter() - val_started
        row: dict[str, object] = {
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "train_seconds": train_seconds,
            "val_seconds": val_seconds,
        }
        for variant in VARIANTS:
            row[variant] = {
                "train": train_metrics[variant],
                "val": val_metrics[variant],
                "lr": float(optimizers[variant].param_groups[0]["lr"]),
            }
            val_loss = val_metrics[variant]["loss"]
            previous = best_metrics.get(variant)
            if previous is None or val_loss < float(previous["val_loss"]):
                best_metrics[variant] = {
                    "epoch": epoch,
                    "optimizer_steps": optimizer_steps,
                    "val_loss": val_loss,
                    "selection_role": "independent_validation_best",
                }
                best_states[variant] = checkpoint_model_state(models[variant])
                epochs_without_improvement[variant] = 0
            else:
                epochs_without_improvement[variant] += 1
        history.append(row)
        completed_epochs = epoch
        print(
            f"epoch {epoch:03d} steps={optimizer_steps} "
            f"residual_train={train_metrics['policy_only_residual']['loss']:.5f} "
            f"residual_val={val_metrics['policy_only_residual']['loss']:.5f} "
            f"deform_train={train_metrics['fk_deform_attn']['loss']:.5f} "
            f"deform_val={val_metrics['fk_deform_attn']['loss']:.5f} "
            f"train_s={train_seconds:.2f} val_s={val_seconds:.2f}",
            flush=True,
        )
        if args.patience > 0 and all(
            value >= args.patience for value in epochs_without_improvement.values()
        ):
            break

    if set(best_states) != set(VARIANTS):
        raise RuntimeError("training completed without selecting both pair members")
    final_states = {
        variant: checkpoint_model_state(model) for variant, model in models.items()
    }
    final_metrics = {
        variant: {
            "epoch": int(completed_epochs),
            "optimizer_steps": int(optimizer_steps),
            "val_loss": float(val_metrics[variant]["loss"]),
            "selection_role": "final_at_matched_budget",
        }
        for variant in VARIANTS
    }
    pair_run_contract = {
        "pair_run_id": pair_run_id,
        "selection_policy": SELECTION_POLICY,
        "maximum_budget": maximum_budget,
        "realized_budget": {
            "epochs": int(completed_epochs),
            "optimizer_steps_per_variant": int(optimizer_steps),
        },
        "shared_training_run": True,
        "paired_batch_order": True,
        "paired_dropout_rng": True,
    }
    example_batch = _single_item(next(iter(val_loader)), device)
    latency: dict[str, dict[str, dict[str, object]]] = {}
    for variant in VARIANTS:
        latency[variant] = {}
    for state_name, states in (("best", best_states), ("final", final_states)):
        for variant in VARIANTS:
            _load_active_state(models[variant], states[variant])
        state_latency = benchmark_interleaved_latency(
            models,
            example_batch,
            device,
            warmup=args.latency_warmup,
            iterations=args.latency_iterations,
            amp=args.amp,
            comparison_id=f"{pair_run_id}:{state_name}",
        )
        for variant in VARIANTS:
            latency[variant][state_name] = state_latency[variant]

    training_contract = {
        "optimizer": "AdamW",
        "optimizer_kwargs": {
            "lr": float(args.lr),
            "betas": (0.95, 0.999),
            "eps": 1e-8,
            "weight_decay": float(args.weight_decay),
        },
        "scheduler": "cosine_with_linear_warmup",
        "warmup_steps": int(args.warmup_steps),
        "planned_epochs": int(args.epochs),
        "completed_epochs": int(completed_epochs),
        "optimizer_steps_per_variant": int(optimizer_steps),
        "batch_size": int(args.batch_size),
        "effective_batch_size": int(args.batch_size),
        "microbatch_size": int(args.microbatch_size or args.batch_size),
        "num_workers": int(args.num_workers),
        "prefetch_factor": (
            int(args.prefetch_factor) if args.num_workers > 0 else None
        ),
        "gradient_accumulation_steps": int(
            math.ceil(args.batch_size / float(args.microbatch_size or args.batch_size))
        ),
        "grad_clip": float(args.grad_clip),
        "amp": bool(args.amp),
        "amp_dtype": "bfloat16" if args.amp else None,
        "gripper_weight": float(args.gripper_weight),
        "paired_batch_order": True,
        "paired_dropout_rng": True,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, dict[str, dict[str, object]]] = {}
    paths: dict[str, dict[str, str]] = {}
    for variant in VARIANTS:
        payloads[variant] = {}
        paths[variant] = {}
        for state_name, state in (
            ("best", best_states[variant]),
            ("final", final_states[variant]),
        ):
            payload = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "checkpoint_type": CHECKPOINT_TYPE,
                "variant": variant,
                "loss_contract": LOSS_CONTRACT,
                "structural_attribution_contract": STRUCTURAL_ATTRIBUTION_CONTRACT,
                "build_contract": build_contracts[variant],
                "parameter_counts": counts[variant],
                "normalization": stats.as_torch(),
                "data_contract": data_contract,
                "vision_cache_validation": vision_cache_validation,
                "training_contract": training_contract,
                "selection_contract": selection_contract,
                "pair_run_contract": pair_run_contract,
                "claim_contract": claim_contract,
                "runtime_budget": {
                    "sampling": build_contracts[variant]["sampling"],
                    "parameters": counts[variant],
                    "matched_maximum_training_budget": maximum_budget,
                    "structural_latency": latency[variant][state_name],
                },
                "checkpoint_state": state_name,
                "selected": (
                    best_metrics[variant]
                    if state_name == "best"
                    else final_metrics[variant]
                ),
                "model_state": state,
            }
            payloads[variant][state_name] = payload
            path = output_dir / f"{variant}_{state_name}.pt"
            atomic_torch_save(payload, path)
            paths[variant][state_name] = str(path.resolve())
    pair_validation_by_state = {
        state_name: validate_structural_pair(
            payloads["policy_only_residual"][state_name],
            payloads["fk_deform_attn"][state_name],
        )
        for state_name in ("best", "final")
    }
    pair_validation = pair_validation_by_state["best"]
    report = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "complete": True,
        "loss_contract": LOSS_CONTRACT,
        "structural_attribution_contract": STRUCTURAL_ATTRIBUTION_CONTRACT,
        "pair_validation": pair_validation,
        "pair_validation_by_checkpoint_state": pair_validation_by_state,
        "data_contract": data_contract,
        "vision_cache_validation": vision_cache_validation,
        "training_contract": training_contract,
        "selection_contract": selection_contract,
        "pair_run_contract": pair_run_contract,
        "claim_contract": claim_contract,
        "selection_summary": {
            "interpretation": SELECTION_POLICY,
            "selected_independently_by_variant": True,
            "same_selected_step_claim_supported": False,
            "matched_maximum_budget": maximum_budget,
            "matched_realized_budget": pair_run_contract["realized_budget"],
        },
        "parameter_counts": counts,
        "best": best_metrics,
        "final": final_metrics,
        "structural_latency": latency,
        "checkpoints": paths,
        "history": history,
    }
    report_path = output_dir / "matched_pair_metrics.json"
    atomic_json_write(report, report_path)
    print(f"pair_validation: {json.dumps(pair_validation)}")
    print(f"metrics: {report_path.resolve()}")
    for variant in VARIANTS:
        print(f"{variant}_best: {paths[variant]['best']}")
        print(f"{variant}_final: {paths[variant]['final']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
