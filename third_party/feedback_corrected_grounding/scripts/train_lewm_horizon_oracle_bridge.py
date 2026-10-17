#!/usr/bin/env python3
"""Train the fixed pooled-H oracle anchor-latent policy-token ceiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.action_chunk_policy import load_action_chunk_policy
from fcpg.lewm_anchor_policy_bridge import build_matched_bridge_pair
from fcpg.lewm_horizon_oracle_bridge import (
    PROTOCOL,
    HorizonBridgeAlignedSplit,
    assert_disjoint_horizon_episodes,
    load_horizon_bridge_source_split,
    reconstruct_horizon_policy_tokens,
    save_horizon_aligned_split,
)
from fcpg.lewm_policy_token_bridge import sha256_file, token_metrics
from fcpg.lewm_residual_policy_token_bridge import (
    GeometryResidualPolicyBridge,
    matched_geometry_residual_hidden_dim,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-posterior-cache", required=True)
    parser.add_argument("--validation-posterior-cache", required=True)
    parser.add_argument("--train-policy-cache", required=True)
    parser.add_argument("--validation-policy-cache", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", default="pick-place-v3,reach-v3,push-v3")
    parser.add_argument("--horizons", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--raw-steps-per-decision", type=int, default=5)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--target-batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=15714)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def _csv_strings(value: str) -> tuple[str, ...]:
    rows = tuple(item.strip() for item in value.split(",") if item.strip())
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("comma-separated values must be non-empty and unique")
    return rows


def _csv_ints(value: str) -> tuple[int, ...]:
    rows = tuple(int(item) for item in _csv_strings(value))
    if min(rows) <= 0:
        raise ValueError("horizons must be positive")
    return rows


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _output(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _moments(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(array, dtype=np.float64).reshape(-1, array.shape[-1])
    return values.mean(axis=0), np.maximum(values.std(axis=0), 1.0e-5)


def _configure(
    primary: torch.nn.Module,
    delta_blind: torch.nn.Module,
    geometry: GeometryResidualPolicyBridge,
    split: HorizonBridgeAlignedSplit,
) -> None:
    start = split.start_index
    target = split.target_index
    last_embedding = split.raw_embedding[start]
    target_embedding = split.raw_embedding[target]
    last_token = split.raw_token[start]
    target_token = split.raw_token[target]
    last_xyz = split.raw_xyz[start]
    current_xyz = split.raw_xyz[target]
    context_mean, context_std = _moments(last_embedding)
    latent_mean, latent_std = _moments(target_embedding - last_embedding)
    xyz_mean, xyz_std = _moments(current_xyz - last_xyz)
    _, token_std = _moments(target_token - last_token)
    primary.set_input_normalization(
        context_mean=context_mean,
        context_std=context_std,
        latent_delta_mean=latent_mean,
        latent_delta_std=latent_std,
        xyz_delta_mean=xyz_mean,
        xyz_delta_std=xyz_std,
        token_error_std=token_std,
    )
    delta_blind.set_input_normalization(
        context_mean=context_mean,
        context_std=context_std,
        latent_delta_mean=np.zeros(192),
        latent_delta_std=np.ones(192),
        xyz_delta_mean=xyz_mean,
        xyz_delta_std=xyz_std,
        token_error_std=token_std,
    )
    geometry.set_normalization(
        last_xyz=last_xyz,
        current_xyz=current_xyz,
        token_delta=target_token - last_token,
    )


def _device_data(
    split: HorizonBridgeAlignedSplit, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "embedding": torch.from_numpy(split.raw_embedding).to(device),
        "token": torch.from_numpy(split.raw_token).to(device),
        "xyz": torch.from_numpy(split.raw_xyz).to(device),
        "start": torch.from_numpy(split.start_index).to(device),
        "target": torch.from_numpy(split.target_index).to(device),
    }


def _forward(
    model: torch.nn.Module,
    data: dict[str, torch.Tensor],
    pair_rows: torch.Tensor,
    *,
    delta_blind: bool,
) -> torch.Tensor:
    start = data["start"][pair_rows]
    target = data["target"][pair_rows]
    last_embedding = data["embedding"][start]
    predicted = last_embedding if delta_blind else data["embedding"][target]
    return model(
        predicted,
        last_embedding,
        data["token"][start],
        data["xyz"][target],
        data["xyz"][start],
    )


def _predict(
    model: torch.nn.Module,
    data: dict[str, torch.Tensor],
    *,
    delta_blind: bool,
    batch_size: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, data["start"].shape[0], batch_size):
            pair = torch.arange(
                start,
                min(start + batch_size, data["start"].shape[0]),
                device=data["start"].device,
            )
            rows.append(_forward(model, data, pair, delta_blind=delta_blind).cpu().numpy())
    return np.concatenate(rows, axis=0)


def _predict_geometry(
    model: GeometryResidualPolicyBridge,
    data: dict[str, torch.Tensor],
    *,
    batch_size: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, data["start"].shape[0], batch_size):
            pair = torch.arange(
                offset,
                min(offset + batch_size, data["start"].shape[0]),
                device=data["start"].device,
            )
            start = data["start"][pair]
            target = data["target"][pair]
            rows.append(
                model(data["token"][start], data["xyz"][target], data["xyz"][start])
                .cpu()
                .numpy()
            )
    return np.concatenate(rows, axis=0)


def _metric_bundle(
    prediction: np.ndarray,
    split: HorizonBridgeAlignedSplit,
    target: np.ndarray,
) -> dict[str, object]:
    result: dict[str, object] = {
        "overall": token_metrics(prediction, target, split.pair_task_name),
        "per_horizon": {},
    }
    for horizon in sorted(set(split.horizon.tolist())):
        mask = split.horizon == horizon
        result["per_horizon"][str(horizon)] = token_metrics(
            prediction[mask], target[mask], split.pair_task_name[mask]
        )
    return result


def main() -> None:
    args = parse_args()
    tasks = _csv_strings(args.tasks)
    horizons = _csv_ints(args.horizons)
    if min(
        args.raw_steps_per_decision,
        args.steps,
        args.batch_size,
        args.target_batch_size,
        args.hidden_dim,
        args.log_every,
    ) <= 0:
        raise ValueError("all integer training settings must be positive")
    output = _output(args.output_dir)
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_source = load_horizon_bridge_source_split(
        args.train_posterior_cache,
        args.train_policy_cache,
        tasks=tasks,
        horizons=horizons,
        raw_steps_per_decision=args.raw_steps_per_decision,
    )
    validation_source = load_horizon_bridge_source_split(
        args.validation_posterior_cache,
        args.validation_policy_cache,
        tasks=tasks,
        horizons=horizons,
        raw_steps_per_decision=args.raw_steps_per_decision,
    )
    assert_disjoint_horizon_episodes(train_source, validation_source)
    policy_checkpoint = _resolve(args.policy_checkpoint)
    policy, policy_args = load_action_chunk_policy(policy_checkpoint, device)
    policy.requires_grad_(False)
    train = reconstruct_horizon_policy_tokens(
        policy, train_source, device=device, batch_size=args.target_batch_size
    )
    validation = reconstruct_horizon_policy_tokens(
        policy, validation_source, device=device, batch_size=args.target_batch_size
    )
    assert_disjoint_horizon_episodes(train, validation)
    policy_corrector = _resolve(str(policy_args.get("fcpg_checkpoint", "")))
    policy_provenance = {
        "policy_checkpoint": str(policy_checkpoint),
        "policy_checkpoint_sha256": sha256_file(policy_checkpoint),
        "policy_corrector_checkpoint": str(policy_corrector),
        "policy_corrector_checkpoint_sha256": sha256_file(policy_corrector),
    }
    if policy_provenance["policy_corrector_checkpoint_sha256"] != str(
        train.provenance["policy_corrector_checkpoint_sha256"]
    ):
        raise ValueError("target policy corrector mismatches source cache")
    save_horizon_aligned_split(output / "train_aligned.npz", train)
    save_horizon_aligned_split(output / "validation_aligned.npz", validation)
    del policy
    if device.type == "cuda":
        torch.cuda.empty_cache()

    primary, delta_blind = build_matched_bridge_pair(
        initialization_seed=args.seed,
        hidden_dim=args.hidden_dim,
    )
    primary_parameters = primary.trainable_parameter_count
    geometry_hidden = matched_geometry_residual_hidden_dim(
        target_parameter_count=primary_parameters,
        token_dim=256,
        num_keypoints=3,
    )
    geometry = GeometryResidualPolicyBridge(hidden_dim=geometry_hidden)
    _configure(primary, delta_blind, geometry, train)
    for model in (primary, delta_blind, geometry):
        model.to(device)
    optimizers = [
        torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        for model in (primary, delta_blind, geometry)
    ]
    data = _device_data(train, device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    trace: list[dict[str, float | int]] = []
    for step in range(1, args.steps + 1):
        pair = torch.randint(
            0, train.pair_count, (args.batch_size,), generator=generator
        ).to(device)
        for model, optimizer in zip((primary, delta_blind, geometry), optimizers, strict=True):
            model.train()
            optimizer.zero_grad(set_to_none=True)
        start = data["start"][pair]
        target_index = data["target"][pair]
        last_token = data["token"][start]
        target_token = data["token"][target_index]
        primary_prediction = _forward(primary, data, pair, delta_blind=False)
        blind_prediction = _forward(delta_blind, data, pair, delta_blind=True)
        geometry_prediction = geometry(
            last_token, data["xyz"][target_index], data["xyz"][start]
        )
        target_delta = (target_token - last_token).detach()
        predictions = (primary_prediction, blind_prediction, geometry_prediction)
        scales = (
            primary.token_error_std,
            delta_blind.token_error_std,
            geometry.token_error_std,
        )
        losses = tuple(
            torch.nn.functional.smooth_l1_loss(
                (prediction - last_token) / scale.clamp_min(1.0e-6),
                target_delta / scale.clamp_min(1.0e-6),
                beta=args.smooth_l1_beta,
            )
            for prediction, scale in zip(predictions, scales, strict=True)
        )
        for loss in losses:
            loss.backward()
        for model, optimizer in zip((primary, delta_blind, geometry), optimizers, strict=True):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if step == 1 or step == args.steps or step % args.log_every == 0:
            row = {
                "step": step,
                "oracle_loss": float(losses[0].detach().cpu()),
                "delta_blind_loss": float(losses[1].detach().cpu()),
                "geometry_loss": float(losses[2].detach().cpu()),
            }
            trace.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    validation_data = _device_data(validation, device)
    primary_prediction = _predict(
        primary, validation_data, delta_blind=False, batch_size=args.target_batch_size
    )
    blind_prediction = _predict(
        delta_blind,
        validation_data,
        delta_blind=True,
        batch_size=args.target_batch_size,
    )
    geometry_prediction = _predict_geometry(
        geometry, validation_data, batch_size=args.target_batch_size
    )
    target_token = validation.raw_token[validation.target_index]
    hold_last = validation.raw_token[validation.start_index]
    metric_bundles = {
        "oracle": _metric_bundle(primary_prediction, validation, target_token),
        "delta_blind": _metric_bundle(blind_prediction, validation, target_token),
        "geometry": _metric_bundle(geometry_prediction, validation, target_token),
        "geometry_hold_exact": _metric_bundle(
            validation.geometry_hold_exact_token, validation, target_token
        ),
        "hold_last": _metric_bundle(hold_last, validation, target_token),
    }
    comparisons: dict[str, object] = {}
    for baseline in ("delta_blind", "geometry", "geometry_hold_exact", "hold_last"):
        baseline_mse = float(metric_bundles[baseline]["overall"]["task_macro_token_mse"])
        oracle_mse = float(metric_bundles["oracle"]["overall"]["task_macro_token_mse"])
        by_horizon = {}
        for horizon in horizons:
            key = str(horizon)
            h_oracle = float(
                metric_bundles["oracle"]["per_horizon"][key]["task_macro_token_mse"]
            )
            h_baseline = float(
                metric_bundles[baseline]["per_horizon"][key]["task_macro_token_mse"]
            )
            by_horizon[key] = (h_baseline - h_oracle) / max(h_baseline, 1.0e-12)
        comparisons[baseline] = {
            "overall_relative_mse_improvement": (
                baseline_mse - oracle_mse
            )
            / max(baseline_mse, 1.0e-12),
            "relative_mse_improvement_by_horizon": by_horizon,
            "positive_horizons": sum(value > 0.0 for value in by_horizon.values()),
        }
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "completed_privileged_horizon_interface_ceiling",
        "interpretation_boundary": {
            "uses_privileged_current_measurement_embedding": True,
            "uses_a3_prediction": False,
            "post_hoc_interface_diagnostic_only": True,
            "deployable": False,
            "control_claim": False,
        },
        "config": vars(args),
        "parameter_matching": {
            "oracle": primary_parameters,
            "delta_blind": delta_blind.trainable_parameter_count,
            "geometry": geometry.trainable_parameter_count,
        },
        "train_provenance": train.provenance,
        "validation_provenance": validation.provenance,
        "policy": policy_provenance,
        "metrics": metric_bundles,
        "comparisons": comparisons,
        "training_trace": trace,
    }
    torch.save(
        {
            "checkpoint_type": PROTOCOL,
            "oracle_state": {k: v.detach().cpu() for k, v in primary.state_dict().items()},
            "delta_blind_state": {
                k: v.detach().cpu() for k, v in delta_blind.state_dict().items()
            },
            "geometry_state": {k: v.detach().cpu() for k, v in geometry.state_dict().items()},
            "config": vars(args),
            "comparisons": comparisons,
        },
        output / "bridge_latest.pt",
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output), "comparisons": comparisons}, sort_keys=True))


if __name__ == "__main__":
    main()
