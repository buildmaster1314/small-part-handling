#!/usr/bin/env python3
"""Train a six-state residual branch on a pretrained SO-101 LeWM."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fcpg.so101_state_residual_lewm import (
    StateResidualLeWM,
    TemporalStateResidualLeWM,
)
from module import SIGReg
from scripts.train_so101_lewm import (
    SIGREG_WEIGHT,
    Cache,
    SO101Sequences,
    _group_counts,
    _make_loader,
    _shuffle_actions_within_group,
    build_model,
    episode_splits,
    global_train_action_stats,
    learning_rate,
    parse_exclusions,
)


METRIC_KEYS = (
    "visual_prediction_mse",
    "visual_base_mse",
    "visual_shuffled_action_mse",
    "visual_shuffled_state_mse",
    "visual_copy_mse",
    "state_prediction_mse",
    "state_command_mse",
    "state_copy_mse",
    "state_shuffled_action_mse",
    "state_shuffled_state_mse",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=5120)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--base-lr-scale", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--val-batches", type=int, default=12)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=7301)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--joint-scale", type=float, default=100.0)
    parser.add_argument(
        "--normalization",
        choices=("global-train-zscore", "physical-scale"),
        default="global-train-zscore",
        help="Shared normalization used for both six-state inputs and actions.",
    )
    parser.add_argument("--motion-threshold", type=float, default=0.01)
    parser.add_argument("--state-loss-weight", type=float, default=1.0)
    parser.add_argument("--latent-residual-cap", type=float, default=0.25)
    parser.add_argument("--state-residual-cap", type=float, default=1.0)
    parser.add_argument(
        "--residual-architecture",
        choices=("mlp", "temporal"),
        default="mlp",
    )
    parser.add_argument("--temporal-model-dim", type=int, default=512)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--temporal-heads", type=int, default=8)
    parser.add_argument("--temporal-ffn-dim", type=int, default=1024)
    parser.add_argument("--temporal-max-steps", type=int, default=32)
    parser.add_argument("--exclude-episode", action="append", default=[])
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help=(
            "Initialize the complete state-residual model from a matching checkpoint "
            "without restoring its optimizer or data contract. Use this for bounded "
            "transfer to a new SO-101 dataset."
        ),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def load_pretrained_base(model: torch.nn.Module, path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError("base checkpoint does not contain a state dict")
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise ValueError("base checkpoint did not load exactly")
    sigreg_weight = payload.get("sigreg_weight") if isinstance(payload, dict) else None
    if sigreg_weight is not None and float(sigreg_weight) != SIGREG_WEIGHT:
        raise ValueError("base checkpoint does not use SIGReg=0.02")
    base_contract = (
        payload.get("run_config", {}).get("contract", {})
        if isinstance(payload, dict)
        else {}
    )
    return {
        "path": str(path.resolve()),
        "step": int(payload.get("step", -1)) if isinstance(payload, dict) else -1,
        "loaded_tensors": len(state),
        "protocol": base_contract.get("protocol"),
        "action_normalization": base_contract.get("action_normalization"),
        "action_center": base_contract.get("action_center"),
        "action_scale": base_contract.get("action_scale"),
    }


def load_transfer_checkpoint(
    model: StateResidualLeWM,
    path: Path,
    *,
    expected_protocol: str,
    expected_center: np.ndarray,
    expected_scale: np.ndarray,
) -> dict:
    """Load model weights for a fresh transfer run while rejecting mismatched contracts."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("transfer checkpoint does not contain a model state dict")
    if float(payload.get("sigreg_weight", float("nan"))) != SIGREG_WEIGHT:
        raise ValueError("transfer checkpoint does not use SIGReg=0.02")
    contract = payload.get("run_config", {}).get("contract", {})
    if contract.get("protocol") != expected_protocol:
        raise ValueError(
            "transfer checkpoint protocol differs from requested architecture: "
            f"{contract.get('protocol')!r} != {expected_protocol!r}"
        )
    center = np.asarray(contract.get("normalization_center"), dtype=np.float32)
    scale = np.asarray(contract.get("normalization_scale"), dtype=np.float32)
    if center.shape != (6,) or not np.allclose(center, expected_center, rtol=1e-6, atol=1e-6):
        raise ValueError("transfer checkpoint action center differs from requested run")
    if scale.shape != (6,) or not np.allclose(scale, expected_scale, rtol=1e-6, atol=1e-6):
        raise ValueError("transfer checkpoint action scale differs from requested run")
    missing, unexpected = model.load_state_dict(payload["model"], strict=True)
    if missing or unexpected:
        raise ValueError("transfer checkpoint did not load exactly")
    return {
        "path": str(path.resolve()),
        "step": int(payload.get("step", -1)),
        "protocol": contract.get("protocol"),
        "loaded_tensors": len(payload["model"]),
    }


def state_action_normalization(
    mode: str,
    joint_scale: float,
    caches: list[Cache],
    assignments: dict[tuple[str, int], str],
    exclusions: set[tuple[str, int]],
) -> tuple[np.ndarray, np.ndarray, int]:
    if mode == "physical-scale":
        if not np.isfinite(joint_scale) or joint_scale <= 0:
            raise ValueError("physical-scale normalization requires positive joint_scale")
        return (
            np.zeros(6, dtype=np.float32),
            np.full(6, joint_scale, dtype=np.float32),
            0,
        )
    if mode == "global-train-zscore":
        return global_train_action_stats(caches, assignments, exclusions)
    raise ValueError(f"unsupported normalization mode {mode!r}")


def validate_base_normalization(
    report: dict,
    mode: str,
    center: np.ndarray,
    scale: np.ndarray,
) -> None:
    if report.get("action_normalization") != mode:
        raise ValueError(
            "base checkpoint action normalization differs from residual training: "
            f"{report.get('action_normalization')!r} != {mode!r}"
        )
    base_center = np.asarray(report.get("action_center"), dtype=np.float32)
    base_scale = np.asarray(report.get("action_scale"), dtype=np.float32)
    if base_center.shape != (6,) or base_scale.shape != (6,):
        raise ValueError("base checkpoint lacks six-dimensional action normalization")
    if not np.allclose(base_center, center, rtol=1e-6, atol=1e-6):
        raise ValueError("base checkpoint action center differs from residual training")
    if not np.allclose(base_scale, scale, rtol=1e-6, atol=1e-6):
        raise ValueError("base checkpoint action scale differs from residual training")


def normalize_state(
    state: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    if state.shape[-1] != 6 or center.shape != (6,) or scale.shape != (6,):
        raise ValueError("SO-101 state normalization expects six dimensions")
    return (state - center) / scale


def set_training_mode(model: StateResidualLeWM, *, train_base: bool) -> None:
    """Train the residual branch without mutating a frozen base's running state."""

    model.train()
    if not train_base:
        model.base.eval()


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    result = {key: float(np.mean([row[key] for row in rows])) for key in METRIC_KEYS}
    result.update(
        {
            "visual_residual_gain": result["visual_base_mse"]
            - result["visual_prediction_mse"],
            "visual_action_delta": result["visual_shuffled_action_mse"]
            - result["visual_prediction_mse"],
            "visual_state_delta": result["visual_shuffled_state_mse"]
            - result["visual_prediction_mse"],
            "visual_copy_gain": result["visual_copy_mse"]
            - result["visual_prediction_mse"],
            "state_command_gain": result["state_command_mse"]
            - result["state_prediction_mse"],
            "state_copy_gain": result["state_copy_mse"]
            - result["state_prediction_mse"],
            "state_action_delta": result["state_shuffled_action_mse"]
            - result["state_prediction_mse"],
            "state_input_delta": result["state_shuffled_state_mse"]
            - result["state_prediction_mse"],
        }
    )
    return result


@torch.no_grad()
def evaluate(
    model: StateResidualLeWM,
    loader: DataLoader,
    device: torch.device,
    group_labels: dict[int, str],
    state_center: torch.Tensor,
    state_scale: torch.Tensor,
) -> dict:
    model.eval()
    all_rows: list[dict[str, float]] = []
    per_group: dict[int, list[dict[str, float]]] = defaultdict(list)
    per_motion: dict[bool, list[dict[str, float]]] = defaultdict(list)
    for batch in loader:
        pixels = batch["pixels"].to(device, non_blocking=True)
        actions = batch["action"].to(device, non_blocking=True)
        states = normalize_state(
            batch["state"].to(device, non_blocking=True), state_center, state_scale
        )
        groups = batch["group"]
        moving = batch["moving"]
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            output = model(pixels, actions, states)
            shuffled_actions = _shuffle_actions_within_group(actions, groups)
            _, shuffled_action_prediction, _, shuffled_action_state = (
                model.predict_from_embedding(
                    output.embedding,
                    shuffled_actions,
                    states[:, :-1],
                )
            )
            shuffled_states = _shuffle_actions_within_group(states[:, :-1], groups)
            _, shuffled_state_prediction, _, shuffled_state_state = (
                model.predict_from_embedding(
                    output.embedding,
                    actions,
                    shuffled_states,
                )
            )
            values = {
                "visual_prediction_mse": (output.prediction - output.target_embedding)
                .square()
                .mean(dim=(1, 2)),
                "visual_base_mse": (output.base_prediction - output.target_embedding)
                .square()
                .mean(dim=(1, 2)),
                "visual_shuffled_action_mse": (
                    shuffled_action_prediction - output.target_embedding
                )
                .square()
                .mean(dim=(1, 2)),
                "visual_shuffled_state_mse": (
                    shuffled_state_prediction - output.target_embedding
                )
                .square()
                .mean(dim=(1, 2)),
                "visual_copy_mse": (
                    output.embedding[:, :-1] - output.target_embedding
                )
                .square()
                .mean(dim=(1, 2)),
                "state_prediction_mse": (output.state_prediction - output.target_state)
                .square()
                .mean(dim=(1, 2)),
                "state_command_mse": (output.command_state - output.target_state)
                .square()
                .mean(dim=(1, 2)),
                "state_copy_mse": (states[:, :-1] - output.target_state)
                .square()
                .mean(dim=(1, 2)),
                "state_shuffled_action_mse": (
                    shuffled_action_state - output.target_state
                )
                .square()
                .mean(dim=(1, 2)),
                "state_shuffled_state_mse": (
                    shuffled_state_state - output.target_state
                )
                .square()
                .mean(dim=(1, 2)),
            }
        for index in range(len(pixels)):
            row = {key: float(value[index]) for key, value in values.items()}
            group = int(groups[index])
            motion = bool(moving[index])
            all_rows.append(row)
            per_group[group].append(row)
            per_motion[motion].append(row)

    if not all_rows:
        raise RuntimeError("validation loader produced no batches")
    by_group = {
        group_labels[group]: {**_mean(rows), "sequences": len(rows)}
        for group, rows in sorted(per_group.items())
    }
    macro_rows = [
        {key: metrics[key] for key in METRIC_KEYS} for metrics in by_group.values()
    ]
    return {
        **_mean(macro_rows),
        "micro": {**_mean(all_rows), "sequences": len(all_rows)},
        "active_motion": {
            **_mean(per_motion.get(True, [])),
            "sequences": len(per_motion.get(True, [])),
        },
        "near_static": {
            **_mean(per_motion.get(False, [])),
            "sequences": len(per_motion.get(False, [])),
        },
        "by_source_task": by_group,
        "macro_groups": len(by_group),
    }


def save_checkpoint(
    path: Path,
    model: StateResidualLeWM,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    args: argparse.Namespace,
    run_config: dict,
    metrics: dict,
    device: torch.device,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "sigreg_weight": SIGREG_WEIGHT,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "run_config": run_config,
            "metrics": metrics,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            ),
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.init_checkpoint is not None and args.resume is not None:
        raise ValueError("--init-checkpoint and --resume are mutually exclusive")
    if args.device.startswith("cuda:1"):
        raise ValueError("CUDA1 is forbidden for this project")
    if args.device != "cuda:0" and args.device != "cpu":
        raise ValueError("use CPU for smoke tests or CUDA0 for training")
    if args.history != 3:
        raise ValueError("LeWM history is fixed to three sampled frames")
    if SIGREG_WEIGHT != 0.02:
        raise AssertionError("SO-101 runs must use SIGReg=0.02")
    if args.steps <= 0 or args.batch_size <= 0 or args.state_loss_weight <= 0:
        raise ValueError("training sizes and state loss weight must be positive")
    if not 0.0 <= args.base_lr_scale <= 1.0:
        raise ValueError("base LR scale must be in [0, 1]")
    if args.residual_architecture == "temporal":
        temporal_sizes = (
            args.temporal_model_dim,
            args.temporal_layers,
            args.temporal_heads,
            args.temporal_ffn_dim,
            args.temporal_max_steps,
        )
        if any(value <= 0 for value in temporal_sizes):
            raise ValueError("temporal residual dimensions must be positive")
        if args.temporal_model_dim % args.temporal_heads:
            raise ValueError("temporal model dimension must be divisible by heads")
        if args.temporal_max_steps < args.history:
            raise ValueError("temporal max steps must cover the full history")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    exclusions = parse_exclusions(args.exclude_episode)
    caches = [Cache(root) for root in args.cache]
    assignments = episode_splits(
        caches, exclusions, split_seed=args.split_seed, val_fraction=args.val_fraction
    )
    center_np, scale_np, normalization_frames = state_action_normalization(
        args.normalization,
        args.joint_scale,
        caches,
        assignments,
        exclusions,
    )
    dataset_kwargs = {
        "exclusions": exclusions,
        "split_seed": args.split_seed,
        "val_fraction": args.val_fraction,
        "joint_scale": args.joint_scale,
        "motion_threshold": args.motion_threshold,
        "action_center": center_np,
        "action_scale": scale_np,
    }
    train_set = SO101Sequences(
        args.cache, "train", args.frameskip, args.history, **dataset_kwargs
    )
    val_set = SO101Sequences(
        args.cache, "val", args.frameskip, args.history, **dataset_kwargs
    )

    base = build_model(args.frameskip * 6, args.history)
    init_report = load_pretrained_base(base, args.base_checkpoint)
    validate_base_normalization(
        init_report,
        args.normalization,
        center_np,
        scale_np,
    )
    expected_protocol = (
        "so101-state-residual-lewm-v5-temporal"
        if args.residual_architecture == "temporal"
        else "so101-state-residual-lewm-v4"
    )
    if args.residual_architecture == "temporal":
        model: StateResidualLeWM = TemporalStateResidualLeWM(
            base,
            model_dim=args.temporal_model_dim,
            layers=args.temporal_layers,
            heads=args.temporal_heads,
            ffn_dim=args.temporal_ffn_dim,
            max_steps=args.temporal_max_steps,
            latent_residual_cap=args.latent_residual_cap,
            state_residual_cap=args.state_residual_cap,
        ).to(device)
    else:
        model = StateResidualLeWM(
            base,
            latent_residual_cap=args.latent_residual_cap,
            state_residual_cap=args.state_residual_cap,
        ).to(device)
    transfer_report = None
    if args.init_checkpoint is not None:
        transfer_report = load_transfer_checkpoint(
            model,
            args.init_checkpoint,
            expected_protocol=expected_protocol,
            expected_center=center_np,
            expected_scale=scale_np,
        )
    if args.base_lr_scale == 0.0:
        model.base.requires_grad_(False)
    branch_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("base.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.base.parameters(),
                "lr": args.lr * args.base_lr_scale,
                "lr_scale": args.base_lr_scale,
            },
            {"params": branch_parameters, "lr": args.lr, "lr_scale": 1.0},
        ],
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    sigreg = SIGReg(knots=17, num_proj=1024).to(device)
    state_center = torch.as_tensor(center_np, device=device)
    state_scale = torch.as_tensor(scale_np, device=device)

    contract = {
        "protocol": expected_protocol,
        "caches": [str(path.resolve()) for path in args.cache],
        "cache_datasets": [cache.dataset_id for cache in train_set.caches],
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "val_fraction": args.val_fraction,
        "frameskip": args.frameskip,
        "history": args.history,
        "state_dim": 6,
        "action_dim": args.frameskip * 6,
        "sampling": "source-task-natural",
        "normalization": (
            "shared-physical-scale-for-state-and-action"
            if args.normalization == "physical-scale"
            else "shared-global-training-zscore-for-state-and-action"
        ),
        "normalization_center": center_np.tolist(),
        "normalization_scale": scale_np.tolist(),
        "normalization_frames": normalization_frames,
        "state_loss_weight": args.state_loss_weight,
        "sigreg_weight": SIGREG_WEIGHT,
        "base_lr_scale": args.base_lr_scale,
        "latent_residual_cap": args.latent_residual_cap,
        "state_residual_cap": args.state_residual_cap,
        "exclusions": sorted(args.exclude_episode),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "base_parameters": sum(parameter.numel() for parameter in model.base.parameters()),
        "residual_parameters": sum(parameter.numel() for parameter in branch_parameters),
    }
    if args.residual_architecture == "temporal":
        contract["residual_architecture"] = "temporal"
        contract["temporal_model_dim"] = args.temporal_model_dim
        contract["temporal_layers"] = args.temporal_layers
        contract["temporal_heads"] = args.temporal_heads
        contract["temporal_ffn_dim"] = args.temporal_ffn_dim
        contract["temporal_max_steps"] = args.temporal_max_steps
    run_config = {
        "contract": contract,
        "base_checkpoint": init_report,
        "transfer_checkpoint": transfer_report,
        "train_sequences": len(train_set),
        "val_sequences": len(val_set),
        "train_groups": _group_counts(train_set),
        "val_groups": _group_counts(val_set),
    }

    start_step = 0
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("run_config", {}).get("contract") != contract:
            raise ValueError("resume contract differs from requested run")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint["step"])

    if args.resume is None and args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"fresh run output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(run_config, indent=2, sort_keys=True), flush=True)

    quick_limit = None if args.val_batches == 0 else args.val_batches * args.batch_size
    quick_indices = val_set.balanced_eval_indices(quick_limit)
    full_indices = val_set.balanced_eval_indices(None)
    quick_loader = _make_loader(val_set, quick_indices, args, device, train=False)
    full_loader = _make_loader(val_set, full_indices, args, device, train=False)

    if args.eval_only:
        if checkpoint is None:
            raise ValueError("--eval-only requires --resume")
        loader = full_loader if args.val_batches == 0 else quick_loader
        metrics = evaluate(
            model, loader, device, val_set.group_labels, state_center, state_scale
        )
        (args.output / "evaluation.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return

    generator = torch.Generator().manual_seed(args.seed)
    train_indices = torch.multinomial(
        train_set.balanced_weights("source-task-natural"),
        num_samples=args.steps * args.batch_size,
        replacement=True,
        generator=generator,
    )[start_step * args.batch_size :]
    train_loader = _make_loader(train_set, train_indices, args, device, train=True)
    train_iterator = iter(train_loader)

    if checkpoint is not None:
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"], device)

    metrics_path = args.output / "metrics.jsonl"
    started = time.time()
    last_metrics = evaluate(
        model, quick_loader, device, val_set.group_labels, state_center, state_scale
    )
    with metrics_path.open("a") as stream:
        stream.write(json.dumps({"step": start_step, **last_metrics}, sort_keys=True) + "\n")
    print(json.dumps({"step": start_step, **last_metrics}, sort_keys=True), flush=True)

    set_training_mode(model, train_base=args.base_lr_scale > 0.0)
    for step in range(start_step + 1, args.steps + 1):
        batch = next(train_iterator)
        pixels = batch["pixels"].to(device, non_blocking=True)
        actions = batch["action"].to(device, non_blocking=True)
        states = normalize_state(
            batch["state"].to(device, non_blocking=True), state_center, state_scale
        )
        scheduled_lr = learning_rate(step, args.steps, args.warmup_steps, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = scheduled_lr * group["lr_scale"]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            output = model(pixels, actions, states)
            visual_loss = (output.prediction - output.target_embedding).square().mean()
            state_loss = F.smooth_l1_loss(
                output.state_prediction, output.target_state
            )
            sigreg_loss = sigreg(output.embedding.transpose(0, 1))
            loss = (
                visual_loss
                + args.state_loss_weight * state_loss
                + SIGREG_WEIGHT * sigreg_loss
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % args.eval_every == 0 or step == args.steps:
            loader = full_loader if step == args.steps else quick_loader
            last_metrics = evaluate(
                model, loader, device, val_set.group_labels, state_center, state_scale
            )
            record = {
                "step": step,
                "train_visual_loss": float(visual_loss.detach()),
                "train_state_loss": float(state_loss.detach()),
                "train_sigreg_loss": float(sigreg_loss.detach()),
                "train_loss": float(loss.detach()),
                "lr": scheduled_lr,
                "elapsed_seconds": time.time() - started,
                **last_metrics,
            }
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
            if step == args.steps:
                (args.output / "evaluation.json").write_text(
                    json.dumps(last_metrics, indent=2, sort_keys=True) + "\n"
                )
            set_training_mode(model, train_base=args.base_lr_scale > 0.0)

        if step % args.save_every == 0 or step == args.steps:
            for path in (
                args.output / f"step_{step:06d}.pt",
                args.output / "latest.pt",
            ):
                save_checkpoint(
                    path,
                    model,
                    optimizer,
                    scaler,
                    step,
                    args,
                    run_config,
                    last_metrics,
                    device,
                )


if __name__ == "__main__":
    main()
