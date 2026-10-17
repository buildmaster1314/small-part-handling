#!/usr/bin/env python3
"""Run the fixed 2k residual LeWM-to-policy-token interface pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.action_chunk_policy import load_action_chunk_policy
from fcpg.lewm_anchor_policy_bridge import (
    ResidualAnchorPolicyBridge,
    build_matched_bridge_pair,
)
from fcpg.lewm_policy_token_bridge import sha256_file, token_metrics
from fcpg.lewm_residual_policy_token_bridge import (
    EMBEDDING_SOURCES,
    PROTOCOL,
    GeometryResidualPolicyBridge,
    ResidualBridgeAlignedSplit,
    assert_disjoint_residual_episodes,
    load_residual_bridge_source_split,
    matched_geometry_residual_hidden_dim,
    reconstruct_residual_policy_tokens,
    save_residual_aligned_split,
)


DEFAULT_TASKS = ("pick-place-v3", "reach-v3", "push-v3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-posterior-cache", required=True)
    parser.add_argument("--validation-posterior-cache", required=True)
    parser.add_argument("--train-policy-cache", required=True)
    parser.add_argument("--validation-policy-cache", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument(
        "--embedding-source",
        choices=EMBEDDING_SOURCES,
        default="teacher_forced_factual_prior",
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--target-batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=15712)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _prepare_output(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _tasks(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(result) != len(set(result)):
        raise ValueError("--tasks must be non-empty and unique")
    return result


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_args(args: argparse.Namespace) -> None:
    integers = (
        args.steps,
        args.batch_size,
        args.target_batch_size,
        args.hidden_dim,
        args.log_every,
    )
    if any(value <= 0 for value in integers):
        raise ValueError("steps, batch sizes, hidden dim, and log interval must be positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise ValueError("learning rate must be finite and positive")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        raise ValueError("weight decay must be finite and non-negative")
    if not np.isfinite(args.smooth_l1_beta) or args.smooth_l1_beta <= 0.0:
        raise ValueError("smooth-L1 beta must be finite and positive")


def _moments(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(array, dtype=np.float64).reshape(-1, array.shape[-1])
    return values.mean(axis=0), np.maximum(values.std(axis=0), 1.0e-5)


def _configure_residual_bridge(
    model: ResidualAnchorPolicyBridge,
    split: ResidualBridgeAlignedSplit,
    *,
    use_predicted_delta: bool,
) -> None:
    context_mean, context_std = _moments(split.last_valid_embedding)
    if use_predicted_delta:
        latent_delta = split.predicted_embedding - split.last_valid_embedding
    else:
        latent_delta = np.zeros_like(split.last_valid_embedding)
    delta_mean, delta_std = _moments(latent_delta)
    xyz_mean, xyz_std = _moments(split.current_xyz - split.last_valid_xyz)
    _, token_std = _moments(split.target_token - split.last_valid_token)
    model.set_input_normalization(
        context_mean=context_mean,
        context_std=context_std,
        latent_delta_mean=delta_mean,
        latent_delta_std=delta_std,
        xyz_delta_mean=xyz_mean,
        xyz_delta_std=xyz_std,
        token_error_std=token_std,
    )


def _tensors(
    split: ResidualBridgeAlignedSplit, device: torch.device
) -> dict[str, torch.Tensor]:
    names = (
        "predicted_embedding",
        "last_valid_embedding",
        "last_valid_token",
        "target_token",
        "last_valid_xyz",
        "current_xyz",
    )
    return {
        name: torch.from_numpy(getattr(split, name)).to(
            device=device, dtype=torch.float32
        )
        for name in names
    }


def _predict_residual(
    model: ResidualAnchorPolicyBridge,
    tensors: dict[str, torch.Tensor],
    *,
    use_predicted_delta: bool,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, tensors["target_token"].shape[0], batch_size):
            stop = start + batch_size
            predicted = (
                tensors["predicted_embedding"][start:stop]
                if use_predicted_delta
                else tensors["last_valid_embedding"][start:stop]
            )
            rows.append(
                model(
                    predicted,
                    tensors["last_valid_embedding"][start:stop],
                    tensors["last_valid_token"][start:stop],
                    tensors["current_xyz"][start:stop],
                    tensors["last_valid_xyz"][start:stop],
                )
                .float()
                .cpu()
                .numpy()
            )
    return np.concatenate(rows, axis=0)


def _predict_geometry(
    model: GeometryResidualPolicyBridge,
    tensors: dict[str, torch.Tensor],
    *,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, tensors["target_token"].shape[0], batch_size):
            stop = start + batch_size
            rows.append(
                model(
                    tensors["last_valid_token"][start:stop],
                    tensors["current_xyz"][start:stop],
                    tensors["last_valid_xyz"][start:stop],
                )
                .float()
                .cpu()
                .numpy()
            )
    return np.concatenate(rows, axis=0)


def _cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _train(
    factual: ResidualAnchorPolicyBridge,
    delta_blind: ResidualAnchorPolicyBridge,
    geometry: GeometryResidualPolicyBridge,
    train: ResidualBridgeAlignedSplit,
    validation: ResidualBridgeAlignedSplit,
    *,
    device: torch.device,
    steps: int,
    batch_size: int,
    evaluation_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    beta: float,
    seed: int,
    log_every: int,
) -> tuple[list[dict[str, float | int]], dict[str, np.ndarray]]:
    train_tensors = _tensors(train, device)
    validation_tensors = _tensors(validation, device)
    for model in (factual, delta_blind, geometry):
        model.to(device)
    optimizers = [
        torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for model in (factual, delta_blind, geometry)
    ]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1)
    trace: list[dict[str, float | int]] = []
    for step in range(1, steps + 1):
        index = torch.randint(
            0, train.sample_count, (batch_size,), generator=generator
        ).to(device)
        for model, optimizer in zip(
            (factual, delta_blind, geometry), optimizers, strict=True
        ):
            model.train()
            optimizer.zero_grad(set_to_none=True)
        factual_prediction = factual(
            train_tensors["predicted_embedding"][index],
            train_tensors["last_valid_embedding"][index],
            train_tensors["last_valid_token"][index],
            train_tensors["current_xyz"][index],
            train_tensors["last_valid_xyz"][index],
        )
        delta_blind_prediction = delta_blind(
            train_tensors["last_valid_embedding"][index],
            train_tensors["last_valid_embedding"][index],
            train_tensors["last_valid_token"][index],
            train_tensors["current_xyz"][index],
            train_tensors["last_valid_xyz"][index],
        )
        geometry_prediction = geometry(
            train_tensors["last_valid_token"][index],
            train_tensors["current_xyz"][index],
            train_tensors["last_valid_xyz"][index],
        )
        target = train_tensors["target_token"][index]
        last_token = train_tensors["last_valid_token"][index]
        target_delta = (target - last_token).detach()
        scales = (
            factual.token_error_std,
            delta_blind.token_error_std,
            geometry.token_error_std,
        )
        predictions_for_loss = (
            factual_prediction,
            delta_blind_prediction,
            geometry_prediction,
        )
        losses = tuple(
            torch.nn.functional.smooth_l1_loss(
                (prediction - last_token) / scale.clamp_min(1.0e-6),
                target_delta / scale.clamp_min(1.0e-6),
                beta=beta,
            )
            for prediction, scale in zip(predictions_for_loss, scales, strict=True)
        )
        for loss in losses:
            loss.backward()
        for model, optimizer in zip(
            (factual, delta_blind, geometry), optimizers, strict=True
        ):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        if step == 1 or step == steps or step % log_every == 0:
            row = {
                "step": step,
                "primary_loss": float(losses[0].detach().cpu()),
                "delta_blind_loss": float(losses[1].detach().cpu()),
                "geometry_loss": float(losses[2].detach().cpu()),
            }
            trace.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    predictions = {
        "factual_train": _predict_residual(
            factual, train_tensors, use_predicted_delta=True, batch_size=evaluation_batch_size
        ),
        "factual_validation": _predict_residual(
            factual,
            validation_tensors,
            use_predicted_delta=True,
            batch_size=evaluation_batch_size,
        ),
        "delta_blind_train": _predict_residual(
            delta_blind,
            train_tensors,
            use_predicted_delta=False,
            batch_size=evaluation_batch_size,
        ),
        "delta_blind_validation": _predict_residual(
            delta_blind,
            validation_tensors,
            use_predicted_delta=False,
            batch_size=evaluation_batch_size,
        ),
        "geometry_train": _predict_geometry(
            geometry, train_tensors, batch_size=evaluation_batch_size
        ),
        "geometry_validation": _predict_geometry(
            geometry, validation_tensors, batch_size=evaluation_batch_size
        ),
    }
    return trace, predictions


def _validate_policy(
    policy_args: dict[str, object],
    checkpoint: Path,
    train: ResidualBridgeAlignedSplit,
    validation: ResidualBridgeAlignedSplit,
) -> dict[str, str]:
    if str(policy_args.get("mode", "")) != "fcpg":
        raise ValueError("residual target policy must use FCPG grounding")
    corrector = _resolve(str(policy_args.get("fcpg_checkpoint", "")))
    corrector_sha = sha256_file(corrector)
    for label, split in (("train", train), ("validation", validation)):
        if Path(str(split.provenance["policy_corrector_checkpoint"])).resolve() != corrector:
            raise ValueError(f"{label} cache corrector path mismatches policy")
        if str(split.provenance["policy_corrector_checkpoint_sha256"]) != corrector_sha:
            raise ValueError(f"{label} cache corrector SHA mismatches policy")
    return {
        "policy_checkpoint": str(checkpoint),
        "policy_checkpoint_sha256": sha256_file(checkpoint),
        "policy_corrector_checkpoint": str(corrector),
        "policy_corrector_checkpoint_sha256": corrector_sha,
    }


def main() -> None:
    args = parse_args()
    _validate_args(args)
    tasks = _tasks(args.tasks)
    output = _prepare_output(args.output_dir)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _seed(args.seed)
    train_source = load_residual_bridge_source_split(
        args.train_posterior_cache,
        args.train_policy_cache,
        tasks=tasks,
        embedding_source=args.embedding_source,
    )
    validation_source = load_residual_bridge_source_split(
        args.validation_posterior_cache,
        args.validation_policy_cache,
        tasks=tasks,
        embedding_source=args.embedding_source,
    )
    assert_disjoint_residual_episodes(train_source, validation_source)
    policy_checkpoint = _resolve(args.policy_checkpoint)
    policy, policy_args = load_action_chunk_policy(policy_checkpoint, device)
    policy.requires_grad_(False)
    train = reconstruct_residual_policy_tokens(
        policy, train_source, device=device, batch_size=args.target_batch_size
    )
    validation = reconstruct_residual_policy_tokens(
        policy, validation_source, device=device, batch_size=args.target_batch_size
    )
    policy_provenance = _validate_policy(
        policy_args, policy_checkpoint, train, validation
    )
    assert_disjoint_residual_episodes(train, validation)
    save_residual_aligned_split(output / "train_aligned.npz", train)
    save_residual_aligned_split(output / "validation_aligned.npz", validation)
    del policy
    if device.type == "cuda":
        torch.cuda.empty_cache()

    factual, delta_blind = build_matched_bridge_pair(
        initialization_seed=args.seed,
        latent_dim=train.predicted_embedding.shape[-1],
        token_dim=train.target_token.shape[-1],
        num_keypoints=train.target_token.shape[1],
        hidden_dim=args.hidden_dim,
    )
    _configure_residual_bridge(factual, train, use_predicted_delta=True)
    _configure_residual_bridge(delta_blind, train, use_predicted_delta=False)
    factual_parameters = factual.trainable_parameter_count
    geometry_hidden = matched_geometry_residual_hidden_dim(
        target_parameter_count=factual_parameters,
        token_dim=train.target_token.shape[-1],
        num_keypoints=train.target_token.shape[1],
    )
    geometry = GeometryResidualPolicyBridge(
        token_dim=train.target_token.shape[-1],
        num_keypoints=train.target_token.shape[1],
        hidden_dim=geometry_hidden,
    )
    geometry.set_normalization(
        last_xyz=train.last_valid_xyz,
        current_xyz=train.current_xyz,
        token_delta=train.target_token - train.last_valid_token,
    )
    trace, predictions = _train(
        factual,
        delta_blind,
        geometry,
        train,
        validation,
        device=device,
        steps=args.steps,
        batch_size=args.batch_size,
        evaluation_batch_size=args.target_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta=args.smooth_l1_beta,
        seed=args.seed,
        log_every=args.log_every,
    )
    primary_train_key = f"{args.embedding_source}_train"
    primary_validation_key = f"{args.embedding_source}_validation"
    metrics = {
        primary_train_key: token_metrics(
            predictions["factual_train"], train.target_token, train.task_name
        ),
        primary_validation_key: token_metrics(
            predictions["factual_validation"], validation.target_token, validation.task_name
        ),
        "delta_blind_train": token_metrics(
            predictions["delta_blind_train"], train.target_token, train.task_name
        ),
        "delta_blind_validation": token_metrics(
            predictions["delta_blind_validation"], validation.target_token, validation.task_name
        ),
        "geometry_train": token_metrics(
            predictions["geometry_train"], train.target_token, train.task_name
        ),
        "geometry_validation": token_metrics(
            predictions["geometry_validation"], validation.target_token, validation.task_name
        ),
        "hold_last_validation": token_metrics(
            validation.last_valid_token, validation.target_token, validation.task_name
        ),
        "geometry_hold_exact_validation": token_metrics(
            validation.geometry_hold_exact_token,
            validation.target_token,
            validation.task_name,
        ),
    }
    factual_mse = float(metrics[primary_validation_key]["task_macro_token_mse"])
    comparisons: dict[str, Any] = {}
    all_task_positive = True
    for baseline in (
        "delta_blind",
        "geometry",
        "hold_last",
        "geometry_hold_exact",
    ):
        baseline_metrics = metrics[f"{baseline}_validation"]
        baseline_mse = float(baseline_metrics["task_macro_token_mse"])
        positive = sum(
            float(metrics[primary_validation_key]["per_task"][task]["token_mse"])
            < float(baseline_metrics["per_task"][task]["token_mse"])
            for task in tasks
        )
        comparisons[baseline] = {
            "relative_task_macro_mse_improvement": (
                baseline_mse - factual_mse
            )
            / max(baseline_mse, 1.0e-12),
            "positive_tasks": positive,
        }
        all_task_positive = all_task_positive and positive == len(tasks)
    diagnostic = {
        "embedding_source": args.embedding_source,
        "comparisons": comparisons,
        "task_count": len(tasks),
        "residual_interface_signal": bool(
            all(
                row["relative_task_macro_mse_improvement"] > 0.0
                for row in comparisons.values()
            )
            and all_task_positive
        ),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "completed_residual_interface_pilot",
        "interpretation_boundary": {
            "teacher_forced_2k_prior": (
                args.embedding_source == "teacher_forced_factual_prior"
            ),
            "embedding_source": args.embedding_source,
            "privileged_oracle_current": args.embedding_source == "oracle_current",
            "a3_10k_recursive_prediction": False,
            "known_valid_last_token_base": True,
            "delta_blind_is_not_no_action_dynamics": True,
            "training_target": "native_policy_token_delta_from_last_valid",
            "representation_interface_only": True,
            "deployable_bridge": False,
            "control_improvement": False,
        },
        "config": {
            "tasks": list(tasks),
            "embedding_source": args.embedding_source,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "target_batch_size": args.target_batch_size,
            "hidden_dim": args.hidden_dim,
            "geometry_hidden_dim": geometry_hidden,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "smooth_l1_beta": args.smooth_l1_beta,
            "seed": args.seed,
            "device": str(device),
            "fixed_endpoint": True,
            "checkpoint_selection": "none",
        },
        "parameter_matching": {
            "primary": factual_parameters,
            "delta_blind": delta_blind.trainable_parameter_count,
            "geometry": geometry.trainable_parameter_count,
            "geometry_relative_difference": abs(
                geometry.trainable_parameter_count - factual_parameters
            )
            / factual_parameters,
        },
        "policy": policy_provenance,
        "train_provenance": train.provenance,
        "validation_provenance": validation.provenance,
        "train_physical_episode_count": len(set(train.physical_episode_sha256.tolist())),
        "validation_physical_episode_count": len(
            set(validation.physical_episode_sha256.tolist())
        ),
        "metrics": metrics,
        "diagnostic": diagnostic,
        "training_trace": trace,
    }
    torch.save(
        {
            "schema_version": 1,
            "checkpoint_type": PROTOCOL,
            "primary_state": _cpu_state(factual),
            "delta_blind_state": _cpu_state(delta_blind),
            "geometry_state": _cpu_state(geometry),
            "config": report["config"],
            "policy": policy_provenance,
            "metrics": metrics,
            "diagnostic": diagnostic,
        },
        output / "bridge_latest.pt",
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output), **diagnostic}, sort_keys=True))


if __name__ == "__main__":
    main()
