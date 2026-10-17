#!/usr/bin/env python3
"""Train the strict H1-H8 A3-to-P1 residual policy-token bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_anchor_policy_bridge import ResidualAnchorPolicyBridge
from fcpg.lewm_formal_policy_bridge import (
    FormalBridgeSplit,
    assert_formal_split_pair,
    build_formal_bridge_checkpoint,
    episode_horizon_row_weights,
    formal_episode_mse,
    formal_token_metrics,
    forward_bridge_replicates,
    load_formal_bridge_split,
    make_formal_bridge_triplet,
    paired_episode_bootstrap,
)
from fcpg.lewm_policy_token_bridge import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--evaluation-batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=15715)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _output_dir(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_args(args: argparse.Namespace) -> None:
    integers = (
        args.steps,
        args.batch_size,
        args.evaluation_batch_size,
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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _moments(
    value: np.ndarray,
    *,
    row_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Episode/horizon-weighted moments with equal anchor/replicate mass."""

    array = np.asarray(value, dtype=np.float64)
    weights = np.asarray(row_weights, dtype=np.float64)
    if array.ndim < 2 or array.shape[0] != weights.shape[0]:
        raise ValueError("normalization values and row weights do not align")
    if weights.ndim != 1 or not np.isclose(weights.sum(), 1.0):
        raise ValueError("normalization row weights must be a unit vector")
    flattened = array.reshape(array.shape[0], -1, array.shape[-1])
    mean = np.sum(weights[:, None] * flattened.mean(axis=1), axis=0)
    variance = np.sum(
        weights[:, None] * np.square(flattened - mean).mean(axis=1), axis=0
    )
    return mean, np.maximum(np.sqrt(variance), 1.0e-5)


def _configure_bridge(
    bridge: ResidualAnchorPolicyBridge,
    split: FormalBridgeSplit,
    predicted: np.ndarray,
) -> None:
    row_weights = episode_horizon_row_weights(split)
    context_mean, context_std = _moments(
        split.last_valid_embedding, row_weights=row_weights
    )
    if predicted.ndim == 4:
        reference = split.last_valid_embedding[:, None]
    elif predicted.ndim == 3:
        reference = split.last_valid_embedding
    else:
        raise ValueError("predicted embedding must be [N,K,D] or [N,R,K,D]")
    delta_mean, delta_std = _moments(
        predicted - reference, row_weights=row_weights
    )
    xyz_mean, xyz_std = _moments(
        split.current_xyz - split.last_valid_xyz, row_weights=row_weights
    )
    _, token_std = _moments(
        split.target_token - split.last_valid_token, row_weights=row_weights
    )
    bridge.set_input_normalization(
        context_mean=context_mean,
        context_std=context_std,
        latent_delta_mean=delta_mean,
        latent_delta_std=delta_std,
        xyz_delta_mean=xyz_mean,
        xyz_delta_std=xyz_std,
        token_error_std=token_std,
    )


def _tensors(split: FormalBridgeSplit, device: torch.device) -> dict[str, torch.Tensor]:
    names = (
        "factual_prediction",
        "matched_no_action_prediction",
        "target_oracle_embedding",
        "last_valid_embedding",
        "last_valid_token",
        "target_token",
        "geometry_hold_token",
        "last_valid_xyz",
        "current_xyz",
    )
    return {
        name: torch.from_numpy(getattr(split, name)).to(device=device, dtype=torch.float32)
        for name in names
    }


def _standardized_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    token_error_std: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Equalize replicates and anchors after independent bridge forwards."""

    if prediction.ndim == 3:
        prediction = prediction[:, None]
    if prediction.ndim != 4 or target.ndim != 3:
        raise ValueError("prediction/target dimensions are invalid")
    if prediction.shape[0] != target.shape[0] or prediction.shape[2:] != target.shape[1:]:
        raise ValueError("prediction and target shapes do not align")
    scale = token_error_std.to(prediction).clamp_min(1.0e-6)
    target_delta = target[:, None].expand_as(prediction).detach()
    element = F.smooth_l1_loss(
        prediction / scale,
        target_delta / scale,
        beta=float(beta),
        reduction="none",
    )
    # token dimensions -> anchors -> correlated A3 replicates -> balanced rows
    return element.mean(dim=-1).mean(dim=-1).mean(dim=-1).mean()


def _predict_replicates(
    bridge: ResidualAnchorPolicyBridge,
    tensors: dict[str, torch.Tensor],
    source: str,
    *,
    batch_size: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    bridge.eval()
    with torch.inference_mode():
        for start in range(0, tensors["target_token"].shape[0], batch_size):
            stop = start + batch_size
            rows.append(
                forward_bridge_replicates(
                    bridge,
                    tensors[source][start:stop],
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


def _predict_oracle(
    bridge: ResidualAnchorPolicyBridge,
    tensors: dict[str, torch.Tensor],
    *,
    batch_size: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    bridge.eval()
    with torch.inference_mode():
        for start in range(0, tensors["target_token"].shape[0], batch_size):
            stop = start + batch_size
            rows.append(
                bridge(
                    tensors["target_oracle_embedding"][start:stop],
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


def _relative(reference: float, candidate: float) -> float:
    return float((reference - candidate) / max(reference, 1.0e-12))


def _report(
    split: FormalBridgeSplit,
    *,
    factual: np.ndarray,
    no_action: np.ndarray,
    oracle: np.ndarray,
    bootstrap_replicates: int = 10_000,
) -> dict[str, Any]:
    predictions = {
        "factual": factual,
        "matched_no_action": no_action,
        "target_oracle": oracle,
        "hold_last": split.last_valid_token,
        "geometry_hold_exact": split.geometry_hold_token,
    }
    metrics = {
        name: formal_token_metrics(value, split)
        for name, value in predictions.items()
    }
    episode = {
        name: formal_episode_mse(value, split)
        for name, value in predictions.items()
    }
    comparisons: list[dict[str, Any]] = []
    hold = metrics["hold_last"]["replicates"][0]
    geometry = metrics["geometry_hold_exact"]["replicates"][0]
    oracle_metric = metrics["target_oracle"]["replicates"][0]
    for index, seed in enumerate(split.replicate_seeds):
        factual_metric = metrics["factual"]["replicates"][index]
        no_action_metric = metrics["matched_no_action"]["replicates"][index]
        row = {
            "replicate_index": index,
            "a3_seed": seed,
            "factual_mse": factual_metric["task_macro_mse"],
            "matched_no_action_mse": no_action_metric["task_macro_mse"],
            "hold_last_mse": hold["task_macro_mse"],
            "geometry_hold_exact_mse": geometry["task_macro_mse"],
            "target_oracle_mse": oracle_metric["task_macro_mse"],
        }
        row["factual_vs_no_action_relative_improvement"] = _relative(
            row["matched_no_action_mse"], row["factual_mse"]
        )
        row["factual_vs_hold_relative_improvement"] = _relative(
            row["hold_last_mse"], row["factual_mse"]
        )
        row["factual_vs_geometry_hold_relative_improvement"] = _relative(
            row["geometry_hold_exact_mse"], row["factual_mse"]
        )
        row["per_horizon"] = {}
        for horizon in range(1, 9):
            key = f"H{horizon}"
            factual_h = factual_metric["per_horizon_task_macro_mse"][key]
            no_action_h = no_action_metric["per_horizon_task_macro_mse"][key]
            hold_h = hold["per_horizon_task_macro_mse"][key]
            geometry_h = geometry["per_horizon_task_macro_mse"][key]
            oracle_h = oracle_metric["per_horizon_task_macro_mse"][key]
            row["per_horizon"][key] = {
                "factual_mse": factual_h,
                "matched_no_action_mse": no_action_h,
                "hold_last_mse": hold_h,
                "geometry_hold_exact_mse": geometry_h,
                "target_oracle_mse": oracle_h,
                "factual_vs_no_action_relative_improvement": _relative(
                    no_action_h, factual_h
                ),
                "factual_vs_hold_relative_improvement": _relative(hold_h, factual_h),
                "factual_vs_geometry_hold_relative_improvement": _relative(
                    geometry_h, factual_h
                ),
                "target_oracle_vs_factual_relative_improvement": _relative(
                    factual_h, oracle_h
                ),
            }
        comparisons.append(row)
    screen: dict[str, bool] = {
        "all_three_factual_better_than_matched_no_action": all(
            row["factual_vs_no_action_relative_improvement"] > 0.0
            for row in comparisons
        ),
        "all_three_factual_better_than_hold_last": all(
            row["factual_vs_hold_relative_improvement"] > 0.0
            for row in comparisons
        ),
        "all_three_factual_better_than_geometry_hold": all(
            row["factual_vs_geometry_hold_relative_improvement"] > 0.0
            for row in comparisons
        ),
        "target_oracle_better_than_hold_last": (
            oracle_metric["task_macro_mse"] < hold["task_macro_mse"]
        ),
        "all_three_target_oracle_better_than_factual": all(
            oracle_metric["task_macro_mse"] < row["factual_mse"]
            for row in comparisons
        ),
    }
    factual_episode = episode["factual"]
    no_action_episode = episode["matched_no_action"]
    hold_episode = episode["hold_last"]
    geometry_episode = episode["geometry_hold_exact"]
    oracle_episode = episode["target_oracle"]
    for name, values in episode.items():
        if not np.array_equal(
            values["physical_episode_sha256"],
            factual_episode["physical_episode_sha256"],
        ):
            raise RuntimeError(f"episode alignment changed for {name}")
    correlated_factual = factual_episode["mse"].mean(axis=1)
    correlated_no_action = no_action_episode["mse"].mean(axis=1)
    episode_statistics = {
        "factual_vs_matched_no_action": paired_episode_bootstrap(
            correlated_factual,
            correlated_no_action,
            factual_episode["task_name"],
            seed=19001,
            replicates=bootstrap_replicates,
        ),
        "factual_vs_hold_last": paired_episode_bootstrap(
            correlated_factual,
            hold_episode["mse"][:, 0],
            factual_episode["task_name"],
            seed=19002,
            replicates=bootstrap_replicates,
        ),
        "factual_vs_geometry_hold_exact": paired_episode_bootstrap(
            correlated_factual,
            geometry_episode["mse"][:, 0],
            factual_episode["task_name"],
            seed=19003,
            replicates=bootstrap_replicates,
        ),
        "target_oracle_vs_hold_last": paired_episode_bootstrap(
            oracle_episode["mse"][:, 0],
            hold_episode["mse"][:, 0],
            factual_episode["task_name"],
            seed=19004,
            replicates=bootstrap_replicates,
        ),
        "target_oracle_vs_factual": paired_episode_bootstrap(
            oracle_episode["mse"][:, 0],
            correlated_factual,
            factual_episode["task_name"],
            seed=19005,
            replicates=bootstrap_replicates,
        ),
    }
    per_horizon_statistics: dict[str, Any] = {}
    for horizon_index, horizon in enumerate(range(1, 9)):
        factual_h = factual_episode["horizon_mse"][:, horizon_index].mean(axis=1)
        no_action_h = no_action_episode["horizon_mse"][:, horizon_index].mean(axis=1)
        hold_h = hold_episode["horizon_mse"][:, horizon_index, 0]
        geometry_h = geometry_episode["horizon_mse"][:, horizon_index, 0]
        oracle_h = oracle_episode["horizon_mse"][:, horizon_index, 0]
        key = f"H{horizon}"
        per_horizon_statistics[key] = {
            "factual_vs_matched_no_action": paired_episode_bootstrap(
                factual_h,
                no_action_h,
                factual_episode["task_name"],
                seed=19100 + horizon * 10 + 1,
                replicates=bootstrap_replicates,
            ),
            "factual_vs_hold_last": paired_episode_bootstrap(
                factual_h,
                hold_h,
                factual_episode["task_name"],
                seed=19100 + horizon * 10 + 2,
                replicates=bootstrap_replicates,
            ),
            "factual_vs_geometry_hold_exact": paired_episode_bootstrap(
                factual_h,
                geometry_h,
                factual_episode["task_name"],
                seed=19100 + horizon * 10 + 3,
                replicates=bootstrap_replicates,
            ),
            "target_oracle_vs_factual": paired_episode_bootstrap(
                oracle_h,
                factual_h,
                factual_episode["task_name"],
                seed=19100 + horizon * 10 + 4,
                replicates=bootstrap_replicates,
            ),
        }
    for comparison_key, per_seed_field in (
        ("matched_no_action", "factual_vs_no_action_relative_improvement"),
        ("hold_last", "factual_vs_hold_relative_improvement"),
        ("geometry_hold_exact", "factual_vs_geometry_hold_relative_improvement"),
    ):
        screen[f"every_horizon_all_three_factual_better_than_{comparison_key}"] = all(
            row["per_horizon"][f"H{horizon}"][per_seed_field] > 0.0
            for row in comparisons
            for horizon in range(1, 9)
        )
    screen["overall_episode_ci_lower_positive_vs_all_nonoracle_baselines"] = all(
        episode_statistics[key]["relative_improvement_ci95"][0] > 0.0
        for key in (
            "factual_vs_matched_no_action",
            "factual_vs_hold_last",
            "factual_vs_geometry_hold_exact",
        )
    )
    screen["every_horizon_episode_point_positive_vs_all_nonoracle_baselines"] = all(
        per_horizon_statistics[f"H{horizon}"][key]["relative_improvement"] > 0.0
        for horizon in range(1, 9)
        for key in (
            "factual_vs_matched_no_action",
            "factual_vs_hold_last",
            "factual_vs_geometry_hold_exact",
        )
    )
    return {
        "metrics": metrics,
        "per_a3_seed": comparisons,
        "episode_statistics": episode_statistics,
        "per_horizon_episode_statistics": per_horizon_statistics,
        "screen": screen,
    }


def _train(
    models: tuple[
        ResidualAnchorPolicyBridge,
        ResidualAnchorPolicyBridge,
        ResidualAnchorPolicyBridge,
    ],
    train: FormalBridgeSplit,
    validation: FormalBridgeSplit,
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
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    factual, no_action, oracle = models
    for model in models:
        model.to(device)
    train_tensors = _tensors(train, device)
    validation_tensors = _tensors(validation, device)
    optimizers = tuple(
        torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for model in models
    )
    probabilities = torch.from_numpy(episode_horizon_row_weights(train)).double()
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    trace: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        index_cpu = torch.multinomial(
            probabilities, batch_size, replacement=True, generator=generator
        )
        index = index_cpu.to(device)
        for model, optimizer in zip(models, optimizers, strict=True):
            model.train()
            optimizer.zero_grad(set_to_none=True)
        factual_prediction = forward_bridge_replicates(
            factual,
            train_tensors["factual_prediction"][index],
            train_tensors["last_valid_embedding"][index],
            train_tensors["last_valid_token"][index],
            train_tensors["current_xyz"][index],
            train_tensors["last_valid_xyz"][index],
        )
        no_action_prediction = forward_bridge_replicates(
            no_action,
            train_tensors["matched_no_action_prediction"][index],
            train_tensors["last_valid_embedding"][index],
            train_tensors["last_valid_token"][index],
            train_tensors["current_xyz"][index],
            train_tensors["last_valid_xyz"][index],
        )
        oracle_prediction = oracle(
            train_tensors["target_oracle_embedding"][index],
            train_tensors["last_valid_embedding"][index],
            train_tensors["last_valid_token"][index],
            train_tensors["current_xyz"][index],
            train_tensors["last_valid_xyz"][index],
        )
        target = train_tensors["target_token"][index]
        losses = (
            _standardized_loss(
                factual_prediction,
                target,
                token_error_std=factual.token_error_std,
                beta=beta,
            ),
            _standardized_loss(
                no_action_prediction,
                target,
                token_error_std=no_action.token_error_std,
                beta=beta,
            ),
            _standardized_loss(
                oracle_prediction,
                target,
                token_error_std=oracle.token_error_std,
                beta=beta,
            ),
        )
        for loss in losses:
            loss.backward()
        for model, optimizer in zip(models, optimizers, strict=True):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if step == 1 or step == steps or step % log_every == 0:
            row = {
                "step": step,
                "factual_loss": float(losses[0].detach().cpu()),
                "matched_no_action_loss": float(losses[1].detach().cpu()),
                "target_oracle_loss": float(losses[2].detach().cpu()),
            }
            trace.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    def evaluate(split: FormalBridgeSplit, tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
        return _report(
            split,
            factual=_predict_replicates(
                factual,
                tensors,
                "factual_prediction",
                batch_size=evaluation_batch_size,
            ),
            no_action=_predict_replicates(
                no_action,
                tensors,
                "matched_no_action_prediction",
                batch_size=evaluation_batch_size,
            ),
            oracle=_predict_oracle(
                oracle, tensors, batch_size=evaluation_batch_size
            ),
        )

    return trace, evaluate(train, train_tensors), evaluate(validation, validation_tensors)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    _seed_everything(args.seed)
    output = _output_dir(args.output_dir)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    train = load_formal_bridge_split(_resolve(args.train_cache))
    validation = load_formal_bridge_split(_resolve(args.validation_cache))
    assert_formal_split_pair(train, validation)
    models = make_formal_bridge_triplet(
        initialization_seed=args.seed,
        hidden_dim=args.hidden_dim,
    )
    _configure_bridge(models[0], train, train.factual_prediction)
    _configure_bridge(models[1], train, train.matched_no_action_prediction)
    _configure_bridge(models[2], train, train.target_oracle_embedding)
    training_config = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "evaluation_batch_size": args.evaluation_batch_size,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "smooth_l1_beta": args.smooth_l1_beta,
        "seed": args.seed,
        "sampling": "episode_then_horizon_equal_in_expectation",
        "anchor_reduction": "equal_mean",
        "a3_replicate_reduction": "independent_forward_then_equal_mean_loss",
        "deployment_semantics": "one_bound_a3_seed_at_a_time",
    }
    trace, train_report, validation_report = _train(
        models,
        train,
        validation,
        device=device,
        steps=args.steps,
        batch_size=args.batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta=args.smooth_l1_beta,
        seed=args.seed,
        log_every=args.log_every,
    )
    report = {
        "protocol": "formal_h1_h8_anchor_policy_token_bridge_v1",
        "train": train_report,
        "validation": validation_report,
        "formal_screen_passed": all(validation_report["screen"].values()),
        "scientific_unit": "physical_episode",
        "a3_seeds_are_correlated_replicates": True,
    }
    checkpoint = build_formal_bridge_checkpoint(
        factual=models[0],
        no_action=models[1],
        target_oracle=models[2],
        train=train,
        validation=validation,
        training_config=training_config,
        report=report,
    )
    checkpoint_path = output / "formal_anchor_policy_bridge.pt"
    torch.save(checkpoint, checkpoint_path)
    report["checkpoint"] = {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
    }
    (output / "training_trace.json").write_text(
        json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["validation"]["screen"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
