#!/usr/bin/env python3
"""Train the bounded Anchor-Persistent-to-policy-token interface pilot."""

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
from fcpg.lewm_policy_token_bridge import (
    PROTOCOL,
    BridgeAlignedSplit,
    PolicyTokenBridge,
    assert_disjoint_physical_episodes,
    bridge_training_loss,
    load_bridge_source_split,
    matched_geometry_hidden_dim,
    reconstruct_native_policy_targets,
    save_aligned_split,
    sha256_file,
    token_metrics,
    trainable_parameter_count,
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
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated task names; fixed order is preserved in provenance.",
    )
    parser.add_argument(
        "--input-key",
        default="teacher_forced_factual_prior",
        choices=("teacher_forced_factual_prior",),
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--target-batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=15711)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _parse_tasks(value: str) -> tuple[str, ...]:
    tasks = tuple(part.strip() for part in value.split(",") if part.strip())
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("--tasks must contain unique non-empty task names")
    return tasks


def _prepare_output(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_args(args: argparse.Namespace) -> None:
    integer_values = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "target_batch_size": args.target_batch_size,
        "hidden_dim": args.hidden_dim,
        "log_every": args.log_every,
    }
    if any(int(value) <= 0 for value in integer_values.values()):
        raise ValueError(f"positive integers required, got {integer_values}")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise ValueError("learning rate must be finite and positive")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        raise ValueError("weight decay must be finite and non-negative")
    if not np.isfinite(args.cosine_weight) or args.cosine_weight < 0.0:
        raise ValueError("cosine weight must be finite and non-negative")


def _validate_policy_provenance(
    policy_args: dict[str, object],
    policy_checkpoint: Path,
    train: BridgeAlignedSplit,
    validation: BridgeAlignedSplit,
) -> dict[str, str]:
    if str(policy_args.get("mode", "")) != "fcpg":
        raise ValueError("bridge target policy must use FCPG grounding")
    policy_corrector = _resolve_repo_path(str(policy_args.get("fcpg_checkpoint", "")))
    if not policy_corrector.is_file():
        raise FileNotFoundError(f"policy corrector is missing: {policy_corrector}")
    corrector_sha256 = sha256_file(policy_corrector)
    for label, split in (("train", train), ("validation", validation)):
        cached_path = Path(
            str(split.provenance["policy_corrector_checkpoint"])
        ).resolve()
        cached_sha256 = str(
            split.provenance["policy_corrector_checkpoint_sha256"]
        )
        if cached_path != policy_corrector or cached_sha256 != corrector_sha256:
            raise ValueError(
                f"{label} policy visual cache does not match target policy corrector"
            )
    return {
        "policy_checkpoint": str(policy_checkpoint),
        "policy_checkpoint_sha256": sha256_file(policy_checkpoint),
        "policy_corrector_checkpoint": str(policy_corrector),
        "policy_corrector_checkpoint_sha256": corrector_sha256,
    }


def _as_device_tensors(
    split: BridgeAlignedSplit, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(split.state).to(device=device, dtype=torch.float32),
        torch.from_numpy(split.geometry).to(device=device, dtype=torch.float32),
        torch.from_numpy(split.target).to(device=device, dtype=torch.float32),
    )


def _evaluate_model(
    model: PolicyTokenBridge,
    inputs: torch.Tensor,
    *,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    result: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, inputs.shape[0], batch_size):
            result.append(
                model(inputs[start : start + batch_size]).float().cpu().numpy()
            )
    return np.concatenate(result, axis=0).astype(np.float32, copy=False)


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def _train_models(
    main: PolicyTokenBridge,
    geometry: PolicyTokenBridge,
    train: BridgeAlignedSplit,
    validation: BridgeAlignedSplit,
    *,
    device: torch.device,
    steps: int,
    batch_size: int,
    evaluation_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    cosine_weight: float,
    seed: int,
    log_every: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    train_state, train_geometry, train_target = _as_device_tensors(train, device)
    validation_state, validation_geometry, _ = _as_device_tensors(validation, device)
    main.set_normalization(train.state, train.target)
    geometry.set_normalization(train.geometry, train.target)
    main.to(device)
    geometry.to(device)
    main_optimizer = torch.optim.AdamW(
        main.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    geometry_optimizer = torch.optim.AdamW(
        geometry.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1)
    trace: list[dict[str, float | int]] = []

    for step in range(1, steps + 1):
        indices = torch.randint(
            0,
            train.sample_count,
            (batch_size,),
            generator=generator,
        ).to(device)
        target = train_target[indices]
        main.train()
        geometry.train()
        main_optimizer.zero_grad(set_to_none=True)
        geometry_optimizer.zero_grad(set_to_none=True)
        main_loss, main_parts = bridge_training_loss(
            main,
            train_state[indices],
            target,
            cosine_weight=cosine_weight,
        )
        geometry_loss, geometry_parts = bridge_training_loss(
            geometry,
            train_geometry[indices],
            target,
            cosine_weight=cosine_weight,
        )
        main_loss.backward()
        geometry_loss.backward()
        torch.nn.utils.clip_grad_norm_(main.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(geometry.parameters(), max_norm=5.0)
        main_optimizer.step()
        geometry_optimizer.step()
        if step == 1 or step == steps or step % log_every == 0:
            row = {
                "step": step,
                "main_loss": float(main_loss.detach().cpu()),
                "main_normalized_mse": float(
                    main_parts["normalized_mse"].cpu()
                ),
                "main_cosine_distance": float(
                    main_parts["cosine_distance"].cpu()
                ),
                "geometry_loss": float(geometry_loss.detach().cpu()),
                "geometry_normalized_mse": float(
                    geometry_parts["normalized_mse"].cpu()
                ),
                "geometry_cosine_distance": float(
                    geometry_parts["cosine_distance"].cpu()
                ),
            }
            trace.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    predictions = {
        "main_train": _evaluate_model(
            main, train_state, batch_size=evaluation_batch_size
        ),
        "main_validation": _evaluate_model(
            main, validation_state, batch_size=evaluation_batch_size
        ),
        "geometry_train": _evaluate_model(
            geometry, train_geometry, batch_size=evaluation_batch_size
        ),
        "geometry_validation": _evaluate_model(
            geometry, validation_geometry, batch_size=evaluation_batch_size
        ),
    }
    return {"trace": trace}, predictions


def main() -> None:
    args = parse_args()
    _validate_args(args)
    output_dir = _prepare_output(args.output_dir)
    tasks = _parse_tasks(args.tasks)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    _seed_everything(args.seed)

    train_source = load_bridge_source_split(
        args.train_posterior_cache,
        args.train_policy_cache,
        tasks=tasks,
        input_key=args.input_key,
    )
    validation_source = load_bridge_source_split(
        args.validation_posterior_cache,
        args.validation_policy_cache,
        tasks=tasks,
        input_key=args.input_key,
    )
    assert_disjoint_physical_episodes(train_source, validation_source)

    policy_checkpoint = _resolve_repo_path(args.policy_checkpoint)
    if not policy_checkpoint.is_file():
        raise FileNotFoundError(f"policy checkpoint is missing: {policy_checkpoint}")
    policy, policy_args = load_action_chunk_policy(policy_checkpoint, device)
    policy.requires_grad_(False)
    train = reconstruct_native_policy_targets(
        policy,
        train_source,
        device=device,
        batch_size=args.target_batch_size,
    )
    validation = reconstruct_native_policy_targets(
        policy,
        validation_source,
        device=device,
        batch_size=args.target_batch_size,
    )
    policy_provenance = _validate_policy_provenance(
        policy_args, policy_checkpoint, train, validation
    )
    assert_disjoint_physical_episodes(train, validation)
    save_aligned_split(output_dir / "train_aligned.npz", train)
    save_aligned_split(output_dir / "validation_aligned.npz", validation)
    del policy
    if device.type == "cuda":
        torch.cuda.empty_cache()

    main_bridge = PolicyTokenBridge(
        input_dim=train.state.shape[-1],
        output_dim=train.target.shape[-1],
        hidden_dim=args.hidden_dim,
        num_keypoints=train.state.shape[1],
    )
    main_parameters = trainable_parameter_count(main_bridge)
    geometry_hidden = matched_geometry_hidden_dim(
        target_parameter_count=main_parameters,
        output_dim=train.target.shape[-1],
    )
    geometry_bridge = PolicyTokenBridge(
        input_dim=train.geometry.shape[-1],
        output_dim=train.target.shape[-1],
        hidden_dim=geometry_hidden,
        num_keypoints=train.geometry.shape[1],
    )
    geometry_parameters = trainable_parameter_count(geometry_bridge)

    training, predictions = _train_models(
        main_bridge,
        geometry_bridge,
        train,
        validation,
        device=device,
        steps=args.steps,
        batch_size=args.batch_size,
        evaluation_batch_size=args.target_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        cosine_weight=args.cosine_weight,
        seed=args.seed,
        log_every=args.log_every,
    )
    mean_target = train.target.mean(axis=0, keepdims=True)
    mean_validation = np.broadcast_to(mean_target, validation.target.shape)
    metrics = {
        "main_train": token_metrics(
            predictions["main_train"], train.target, train.task_name
        ),
        "main_validation": token_metrics(
            predictions["main_validation"], validation.target, validation.task_name
        ),
        "geometry_train": token_metrics(
            predictions["geometry_train"], train.target, train.task_name
        ),
        "geometry_validation": token_metrics(
            predictions["geometry_validation"],
            validation.target,
            validation.task_name,
        ),
        "mean_validation": token_metrics(
            mean_validation, validation.target, validation.task_name
        ),
    }
    main_mse = float(metrics["main_validation"]["task_macro_token_mse"])
    geometry_mse = float(
        metrics["geometry_validation"]["task_macro_token_mse"]
    )
    mean_mse = float(metrics["mean_validation"]["task_macro_token_mse"])
    relative_improvement = (geometry_mse - main_mse) / max(geometry_mse, 1.0e-12)
    relative_improvement_vs_mean = (mean_mse - main_mse) / max(mean_mse, 1.0e-12)
    positive_tasks = sum(
        float(metrics["main_validation"]["per_task"][task]["token_mse"])
        < float(metrics["geometry_validation"]["per_task"][task]["token_mse"])
        for task in tasks
    )
    positive_tasks_vs_mean = sum(
        float(metrics["main_validation"]["per_task"][task]["token_mse"])
        < float(metrics["mean_validation"]["per_task"][task]["token_mse"])
        for task in tasks
    )
    diagnostic = {
        "relative_task_macro_mse_improvement_vs_geometry": relative_improvement,
        "relative_task_macro_mse_improvement_vs_mean": relative_improvement_vs_mean,
        "positive_tasks_vs_geometry": positive_tasks,
        "positive_tasks_vs_mean": positive_tasks_vs_mean,
        "task_count": len(tasks),
        "interface_signal": bool(
            relative_improvement > 0.0
            and relative_improvement_vs_mean > 0.0
            and positive_tasks == len(tasks)
            and positive_tasks_vs_mean == len(tasks)
        ),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "completed_interface_pilot",
        "interpretation_boundary": {
            "uses_teacher_forced_2k_prior": True,
            "uses_a3_10k_recursive_prediction": False,
            "tests_representation_compatibility_only": True,
            "establishes_deployable_bridge": False,
            "establishes_control_improvement": False,
        },
        "config": {
            "tasks": list(tasks),
            "steps": args.steps,
            "batch_size": args.batch_size,
            "target_batch_size": args.target_batch_size,
            "hidden_dim": args.hidden_dim,
            "geometry_hidden_dim": geometry_hidden,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "cosine_weight": args.cosine_weight,
            "seed": args.seed,
            "device": str(device),
            "fixed_endpoint": True,
            "checkpoint_selection": "none",
        },
        "parameter_matching": {
            "main": main_parameters,
            "geometry": geometry_parameters,
            "relative_difference": abs(main_parameters - geometry_parameters)
            / main_parameters,
        },
        "policy": policy_provenance,
        "train_provenance": train.provenance,
        "validation_provenance": validation.provenance,
        "train_physical_episode_count": len(
            set(train.physical_episode_sha256.tolist())
        ),
        "validation_physical_episode_count": len(
            set(validation.physical_episode_sha256.tolist())
        ),
        "metrics": metrics,
        "diagnostic": diagnostic,
        "training": training,
    }
    torch.save(
        {
            "schema_version": 1,
            "checkpoint_type": PROTOCOL,
            "main_state": _cpu_state_dict(main_bridge),
            "geometry_state": _cpu_state_dict(geometry_bridge),
            "config": report["config"],
            "policy": policy_provenance,
            "metrics": metrics,
            "diagnostic": diagnostic,
        },
        output_dir / "bridge_latest.pt",
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), **diagnostic}, sort_keys=True))


if __name__ == "__main__":
    main()
