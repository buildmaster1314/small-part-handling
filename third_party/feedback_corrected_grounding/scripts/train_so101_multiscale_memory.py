#!/usr/bin/env python3
"""Train a causal multi-scale history adapter on a frozen SO-101 LeWM."""

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

from fcpg.so101_multiscale_memory import (
    MultiScaleHistoryLeWM,
    random_freeze_recent_context_with_validity,
)
from fcpg.so101_state_residual_lewm import TemporalStateResidualLeWM
from module import SIGReg
from scripts.train_so101_lewm import (
    SIGREG_WEIGHT,
    SO101Sequences,
    _group_counts,
    _make_loader,
    _shuffle_actions_within_group,
    learning_rate,
)
from scripts.train_so101_state_residual_lewm import normalize_state, save_checkpoint
from scripts.visualize_so101_lewm import build_checkpoint_model


PROTOCOL = "so101-state-residual-lewm-v6-multiscale-memory"
RELIABILITY_PROTOCOL = "so101-state-residual-lewm-v7-reliability-memory"
METRIC_KEYS = (
    "visual_prediction_mse",
    "visual_backbone_mse",
    "visual_shuffled_memory_mse",
    "state_prediction_mse",
    "state_backbone_mse",
    "state_shuffled_memory_mse",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache",
        type=Path,
        action="append",
        default=[],
        help="Use a parent-listed cache subset for bounded smoke runs.",
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=5130)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--val-batches", type=int, default=12)
    parser.add_argument("--state-loss-weight", type=float, default=1.0)
    parser.add_argument("--memory-offsets", default="5,10,15,20,30,40,60,80,100")
    parser.add_argument("--memory-model-dim", type=int, default=384)
    parser.add_argument("--memory-heads", type=int, default=6)
    parser.add_argument("--memory-latent-cap", type=float, default=0.1)
    parser.add_argument("--memory-state-cap", type=float, default=0.25)
    parser.add_argument(
        "--recent-freeze-probability",
        type=float,
        default=0.0,
        help="Randomly freeze a suffix of the three recent context frames.",
    )
    parser.add_argument(
        "--use-recent-validity",
        action="store_true",
        help="Condition the memory query on a deployable three-frame validity mask.",
    )
    parser.add_argument(
        "--use-fourier-age-encoding",
        action="store_true",
        help="Add multi-frequency causal age encoding to every history token.",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    try:
        args.memory_offsets = tuple(
            int(value.strip())
            for value in args.memory_offsets.split(",")
            if value.strip()
        )
    except ValueError as exc:
        parser.error(f"invalid --memory-offsets: {exc}")
    if not args.memory_offsets:
        parser.error("--memory-offsets cannot be empty")
    if args.device.startswith("cuda:1"):
        parser.error("CUDA1 is forbidden for this project")
    if args.device not in {"cpu", "cuda:0"}:
        parser.error("use CPU or CUDA0")
    if min(args.steps, args.batch_size, args.eval_every, args.save_every) <= 0:
        parser.error("training sizes must be positive")
    if args.val_batches < 0 or args.workers < 0:
        parser.error("val-batches and workers must be non-negative")
    if not 0.0 <= args.recent_freeze_probability <= 1.0:
        parser.error("--recent-freeze-probability must be in [0,1]")
    if SIGREG_WEIGHT != 0.02:
        parser.error("SO-101 memory runs must use SIGReg=0.02")
    return args


def _load_parent(
    path: Path, device: torch.device
) -> tuple[dict, dict, TemporalStateResidualLeWM]:
    path = path.expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("backbone checkpoint does not contain a model state dict")
    if float(payload.get("sigreg_weight", float("nan"))) != SIGREG_WEIGHT:
        raise ValueError("backbone checkpoint does not use SIGReg=0.02")
    contract = payload.get("run_config", {}).get("contract", {})
    if contract.get("protocol") != "so101-state-residual-lewm-v5-temporal":
        raise ValueError("memory adapter requires a v5 temporal state-residual LeWM")
    if int(contract.get("history", 0)) != 3 or int(contract.get("frameskip", 0)) != 5:
        raise ValueError("memory adapter requires the released 3x5 recent-history contract")
    model = build_checkpoint_model(payload, 5, 3, device)
    if not isinstance(model, TemporalStateResidualLeWM):
        raise TypeError("backbone checkpoint did not build a temporal state-residual LeWM")
    return payload, contract, model


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    result = {
        key: float(np.mean([row[key] for row in rows])) for key in METRIC_KEYS
    }
    result.update(
        {
            "visual_memory_gain": result["visual_backbone_mse"]
            - result["visual_prediction_mse"],
            "visual_memory_shuffle_delta": result["visual_shuffled_memory_mse"]
            - result["visual_prediction_mse"],
            "state_memory_gain": result["state_backbone_mse"]
            - result["state_prediction_mse"],
            "state_memory_shuffle_delta": result["state_shuffled_memory_mse"]
            - result["state_prediction_mse"],
        }
    )
    return result


def _to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    center: torch.Tensor,
    scale: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "pixels": batch["pixels"].to(device, non_blocking=True),
        "actions": batch["action"].to(device, non_blocking=True),
        "state": normalize_state(
            batch["state"].to(device, non_blocking=True), center, scale
        ),
        "memory_pixels": batch["memory_pixels"].to(device, non_blocking=True),
        "memory_actions": batch["memory_action"].to(device, non_blocking=True),
        "memory_state": normalize_state(
            batch["memory_state"].to(device, non_blocking=True), center, scale
        ),
        "memory_age": batch["memory_age"].to(device, non_blocking=True),
        "memory_valid": batch["memory_valid"].to(device, non_blocking=True),
    }


def _forward(
    model: MultiScaleHistoryLeWM,
    values: dict[str, torch.Tensor],
):
    return model(
        values["pixels"],
        values["actions"],
        values["state"],
        memory_pixels=values["memory_pixels"],
        memory_actions=values["memory_actions"],
        memory_state=values["memory_state"],
        memory_age=values["memory_age"],
        memory_valid=values["memory_valid"],
        recent_visual_valid=values.get("recent_visual_valid"),
    )


@torch.no_grad()
def evaluate(
    model: MultiScaleHistoryLeWM,
    loader,
    device: torch.device,
    group_labels: dict[int, str],
    center: torch.Tensor,
    scale: torch.Tensor,
) -> dict:
    model.eval()
    all_rows: list[dict[str, float]] = []
    per_group: dict[int, list[dict[str, float]]] = defaultdict(list)
    for batch in loader:
        values = _to_device(batch, device, center, scale)
        groups = batch["group"]
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            output = _forward(model, values)
            shuffled_embedding = _shuffle_actions_within_group(
                output.memory_embedding, groups
            )
            shuffled_actions = _shuffle_actions_within_group(
                values["memory_actions"], groups
            )
            shuffled_state = _shuffle_actions_within_group(
                values["memory_state"], groups
            )
            shuffled_age = _shuffle_actions_within_group(values["memory_age"], groups)
            shuffled_valid = _shuffle_actions_within_group(
                values["memory_valid"], groups
            )
            shuffled = model.predict_context(
                output.embedding[:, :-1],
                values["actions"],
                values["state"][:, :-1],
                memory_embedding=shuffled_embedding,
                memory_actions=shuffled_actions,
                memory_state=shuffled_state,
                memory_age=shuffled_age,
                memory_valid=shuffled_valid,
            )
            metrics = {
                "visual_prediction_mse": (
                    output.prediction - output.target_embedding
                ).square().mean(dim=(1, 2)),
                "visual_backbone_mse": (
                    output.backbone_prediction - output.target_embedding
                ).square().mean(dim=(1, 2)),
                "visual_shuffled_memory_mse": (
                    shuffled[1] - output.target_embedding
                ).square().mean(dim=(1, 2)),
                "state_prediction_mse": (
                    output.state_prediction - output.target_state
                ).square().mean(dim=(1, 2)),
                "state_backbone_mse": (
                    output.backbone_state_prediction - output.target_state
                ).square().mean(dim=(1, 2)),
                "state_shuffled_memory_mse": (
                    shuffled[4] - output.target_state
                ).square().mean(dim=(1, 2)),
            }
        for index in range(len(groups)):
            row = {key: float(value[index]) for key, value in metrics.items()}
            group = int(groups[index])
            all_rows.append(row)
            per_group[group].append(row)
    if not all_rows:
        raise RuntimeError("validation loader produced no batches")
    by_group = {
        group_labels[group]: {**_mean(rows), "sequences": len(rows)}
        for group, rows in sorted(per_group.items())
    }
    macro = _mean(
        [{key: metrics[key] for key in METRIC_KEYS} for metrics in by_group.values()]
    )
    return {
        **macro,
        "micro": {**_mean(all_rows), "sequences": len(all_rows)},
        "by_source_task": by_group,
        "macro_groups": len(by_group),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    parent, parent_contract, backbone = _load_parent(args.backbone_checkpoint, device)
    center_np = np.asarray(parent_contract["normalization_center"], dtype=np.float32)
    scale_np = np.asarray(parent_contract["normalization_scale"], dtype=np.float32)
    parent_caches = [Path(path).expanduser().resolve() for path in parent_contract["caches"]]
    requested_caches = [path.expanduser().resolve() for path in args.cache]
    unknown_caches = sorted(set(requested_caches) - set(parent_caches))
    if unknown_caches:
        raise ValueError(
            "--cache must select only caches already bound to the parent: "
            + ", ".join(str(path) for path in unknown_caches)
        )
    caches = requested_caches or parent_caches
    exclusions = {
        (value.rsplit(":", 1)[0], int(value.rsplit(":", 1)[1]))
        for value in parent_contract.get("exclusions", ())
    }
    dataset_kwargs = {
        "exclusions": exclusions,
        "split_seed": int(parent_contract["split_seed"]),
        "val_fraction": float(parent_contract["val_fraction"]),
        "joint_scale": 100.0,
        "motion_threshold": 0.01,
        "action_center": center_np,
        "action_scale": scale_np,
        "memory_offsets": args.memory_offsets,
    }
    train_set = SO101Sequences(caches, "train", 5, 3, **dataset_kwargs)
    val_set = SO101Sequences(caches, "val", 5, 3, **dataset_kwargs)
    model = MultiScaleHistoryLeWM(
        backbone,
        model_dim=args.memory_model_dim,
        heads=args.memory_heads,
        latent_residual_cap=args.memory_latent_cap,
        state_residual_cap=args.memory_state_cap,
        recent_history=3,
        use_recent_reliability=args.use_recent_validity,
        use_fourier_age_encoding=args.use_fourier_age_encoding,
        freeze_backbone=True,
    ).to(device)
    adapter_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.")
    ]
    optimizer = torch.optim.AdamW(
        adapter_parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    sigreg = SIGReg(knots=17, num_proj=1024).to(device)
    center = torch.as_tensor(center_np, device=device)
    scale = torch.as_tensor(scale_np, device=device)

    contract = {
        "protocol": (
            RELIABILITY_PROTOCOL
            if args.use_recent_validity or args.use_fourier_age_encoding
            else PROTOCOL
        ),
        "parent_protocol": parent_contract["protocol"],
        "parent_checkpoint": str(args.backbone_checkpoint.expanduser().resolve()),
        "caches": [str(path.resolve()) for path in caches],
        "parent_caches": [str(path) for path in parent_caches],
        "cache_subset": bool(requested_caches),
        "exclusions": list(parent_contract.get("exclusions", ())),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "split_seed": parent_contract["split_seed"],
        "val_fraction": parent_contract["val_fraction"],
        "frameskip": 5,
        "history": 3,
        "state_dim": 6,
        "memory_offsets": list(args.memory_offsets),
        "memory_model_dim": args.memory_model_dim,
        "memory_heads": args.memory_heads,
        "memory_latent_cap": args.memory_latent_cap,
        "memory_state_cap": args.memory_state_cap,
        "recent_freeze_probability": args.recent_freeze_probability,
        "use_recent_reliability": args.use_recent_validity,
        "use_fourier_age_encoding": args.use_fourier_age_encoding,
        "recent_history": 3,
        "normalization_center": center_np.tolist(),
        "normalization_scale": scale_np.tolist(),
        "normalization": parent_contract.get("normalization"),
        "state_loss_weight": args.state_loss_weight,
        "sigreg_weight": SIGREG_WEIGHT,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "backbone_parameters": sum(
            parameter.numel() for parameter in model.backbone.parameters()
        ),
        "adapter_parameters": sum(parameter.numel() for parameter in adapter_parameters),
        "backbone_temporal_model_dim": parent_contract["temporal_model_dim"],
        "backbone_temporal_layers": parent_contract["temporal_layers"],
        "backbone_temporal_heads": parent_contract["temporal_heads"],
        "backbone_temporal_ffn_dim": parent_contract["temporal_ffn_dim"],
        "backbone_temporal_max_steps": parent_contract["temporal_max_steps"],
        "backbone_latent_residual_cap": parent_contract["latent_residual_cap"],
        "backbone_state_residual_cap": parent_contract["state_residual_cap"],
    }
    run_config = {
        "contract": contract,
        "parent_step": int(parent.get("step", -1)),
        "train_sequences": len(train_set),
        "val_sequences": len(val_set),
        "train_groups": _group_counts(train_set),
        "val_groups": _group_counts(val_set),
    }

    start_step = 0
    checkpoint = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("run_config", {}).get("contract") != contract:
            raise ValueError("resume contract differs from requested memory run")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint["step"])
    elif args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"fresh run output is not empty: {args.output}")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    quick_limit = None if args.val_batches == 0 else args.val_batches * args.batch_size
    quick_loader = _make_loader(
        val_set, val_set.balanced_eval_indices(quick_limit), args, device, train=False
    )
    full_loader = _make_loader(
        val_set, val_set.balanced_eval_indices(None), args, device, train=False
    )
    if args.eval_only:
        if checkpoint is None:
            raise ValueError("--eval-only requires --resume")
        metrics = evaluate(model, full_loader, device, val_set.group_labels, center, scale)
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
        model, quick_loader, device, val_set.group_labels, center, scale
    )
    metrics_path.write_text(
        json.dumps({"step": start_step, **last_metrics}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        batch = next(train_iterator)
        values = _to_device(batch, device, center, scale)
        if args.recent_freeze_probability > 0:
            values["pixels"], values["recent_visual_valid"] = (
                random_freeze_recent_context_with_validity(
                    values["pixels"],
                    values["memory_pixels"],
                    args.recent_freeze_probability,
                )
            )
        scheduled_lr = learning_rate(step, args.steps, args.warmup_steps, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = scheduled_lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            output = _forward(model, values)
            visual_loss = (
                output.prediction - output.target_embedding
            ).square().mean()
            state_loss = F.smooth_l1_loss(
                output.state_prediction, output.target_state
            )
        # SIGReg's projected statistics overflow under FP16 autocast on the
        # first zero-initialized adapter update.  Keep its prescribed weight,
        # but evaluate the regularizer in FP32 so GradScaler does not skip the
        # entire optimizer step.
        with torch.autocast(
            device_type="cuda", enabled=False
        ):
            sigreg_loss = sigreg(
                output.adapter_embedding.float().transpose(0, 1)
            )
        loss = (
            visual_loss.float()
            + args.state_loss_weight * state_loss.float()
            + SIGREG_WEIGHT * sigreg_loss
        )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(adapter_parameters, 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % args.eval_every == 0 or step == args.steps:
            loader = full_loader if step == args.steps else quick_loader
            last_metrics = evaluate(
                model, loader, device, val_set.group_labels, center, scale
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
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
            if step == args.steps:
                (args.output / "evaluation.json").write_text(
                    json.dumps(last_metrics, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            model.train()

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
