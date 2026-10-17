#!/usr/bin/env python3
"""Jointly fine-tune one shared seven-task visual encoder and action-prefix LeWM."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a4 import HORIZON, SIGREG_WEIGHT, TASKS  # noqa: E402
from fcpg.lewm_action_prefix_joint_a4 import (  # noqa: E402
    ENDPOINT_METRIC_LOSS_WEIGHT,
    GRIPPER_APERTURE_LOSS_WEIGHT,
    JOINT_PROTOCOL,
    LOSS_PROFILE_UNIFORM,
    LOSS_PROFILES,
    METRIC_LOSS_WEIGHT,
    build_joint_model_from_checkpoints,
    input_statistics,
    joint_batch_from_memmaps,
    joint_checkpoint_payload,
    task_balanced_sample,
)
from fcpg.lewm_action_prefix_joint_data import open_raw_replay_arrays  # noqa: E402
from fcpg.lewm_runtime import cosine_warmup_lambda, seed_everything  # noqa: E402


DEFAULT_RAW = ROOT / "outputs/lewm_action_prefix_a4/joint_raw_24train_8dev"
DEFAULT_BASE = ROOT / (
    "outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k/"
    "lewm_grounded_latest.pt"
)
DEFAULT_PREFIX = ROOT / (
    "outputs/lewm_action_prefix_a4/shared_7task_seed40401_geometry10_2k/"
    "lewm_action_prefix_latest.pt"
)
ENCODER_LR = 5.0e-5
BACKBONE_LR = 1.0e-5
HEAD_LR = 3.0e-4
WEIGHT_DECAY = 1.0e-4
WARMUP_STEPS = 100
GRAD_CLIP = 1.0


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--raw-dataset", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--prefix-checkpoint", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=40_501)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--loss-profile",
        choices=LOSS_PROFILES,
        default=LOSS_PROFILE_UNIFORM,
    )
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.seed <= 0:
        parser.error("--steps and --seed must be positive")
    if args.batch_size < len(TASKS):
        parser.error("--batch-size must be at least seven")
    if args.eval_batch_size <= 0 or args.log_every <= 0:
        parser.error("evaluation batch size and log interval must be positive")
    return args


def _cuda0(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("joint A4 training permits only CUDA device 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(value), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _move(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device, non_blocking=True)
        for name, value in batch.items()
    }


def _parameter_groups(model: nn.Module) -> list[dict[str, Any]]:
    vision = getattr(model.frame_encoder, "vision_encoder", None)
    backbone = getattr(vision, "backbone", None)
    backbone_parameters = list(backbone.parameters()) if isinstance(backbone, nn.Module) else []
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    encoder_parameters = [
        parameter
        for module in (model.frame_encoder, model.state_projector)
        for parameter in module.parameters()
        if id(parameter) not in backbone_ids
    ]
    head_parameters = list(model.dynamics.parameters()) + list(model.metric_readout.parameters())
    groups = []
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": BACKBONE_LR, "name": "vit_backbone"})
    groups.append({"params": encoder_parameters, "lr": ENCODER_LR, "name": "frame_encoder"})
    groups.append({"params": head_parameters, "lr": HEAD_LR, "name": "dynamics_and_metric"})
    flattened = [parameter for group in groups for parameter in group["params"]]
    if len({id(parameter) for parameter in flattened}) != len(flattened):
        raise RuntimeError("joint A4 optimizer parameter groups overlap")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if {id(parameter) for parameter in flattened} != {id(parameter) for parameter in trainable}:
        raise RuntimeError("joint A4 optimizer does not cover every trainable parameter")
    return groups


def _decode_metric_for_model(
    model: nn.Module,
    *,
    predicted_state: torch.Tensor,
    current_state: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    image_size: tuple[int, int],
) -> torch.Tensor:
    decode_metric = getattr(model, "decode_metric", None)
    if callable(decode_metric):
        from fcpg.lewm_pose_conditioned_joint_a4 import camera_features_from_batch

        return decode_metric(
            predicted_state=predicted_state,
            current_state=current_state,
            current_q_px=batch["current_q_corrected"],
            current_keypoints_xyz=batch["current_keypoints_xyz"],
            task_index=batch["task_index"],
            camera_features=camera_features_from_batch(batch),
            image_size=image_size,
        )
    return model.metric_readout(
        predicted_state,
        current_state,
        batch["current_q_corrected"],
        batch["task_index"],
        image_size=image_size,
    )


def _predict_dynamics_for_model(
    model: nn.Module,
    *,
    current_state: torch.Tensor,
    future_action: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Run a counterfactual action through the model's deployed action frame."""

    transform = getattr(model, "transform_future_action", None)
    if callable(transform):
        future_action = transform(future_action, batch)
    return model.dynamics(
        current_state,
        future_action.to(dtype=current_state.dtype),
        task_index=batch["task_index"],
    )


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    index: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    development = np.flatnonzero(np.asarray(index["split"]).astype(str) == "development")
    pairs = [(int(bundle), branch) for bundle in development for branch in range(8)]
    totals = {
        task: {
            "count": 0,
            "latent": 0.0,
            "copy": 0.0,
            "shuffle": 0.0,
            "epe": 0.0,
            "cyclic_epe": 0.0,
            "zero_epe": 0.0,
            "hold_epe": 0.0,
            "aperture": 0.0,
            "release_aperture": 0.0,
            "release_count": 0,
            "direction_degrees": 0.0,
            "direction_count": 0,
        }
        for task in TASKS
    }
    view_totals: dict[str, dict[str, float | int]] = {}
    for start in range(0, len(pairs), batch_size):
        current = pairs[start : start + batch_size]
        bundles = np.asarray([value[0] for value in current], dtype=np.int64)
        branches = np.asarray([value[1] for value in current], dtype=np.int64)
        batch = _move(
            joint_batch_from_memmaps(
                index,
                arrays,
                bundle_indices=bundles,
                branch_indices=branches,
            ),
            device,
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(batch)
            cyclic_action = torch.from_numpy(
                np.ascontiguousarray(
                    np.asarray(index["future_action"])[
                        bundles,
                        (branches + 1) % 8,
                    ]
                )
            ).to(device=device, dtype=output.current_state.dtype)
            shuffled_state = _predict_dynamics_for_model(
                model,
                current_state=output.current_state,
                future_action=cyclic_action,
                batch=batch,
            )
            zero_state = _predict_dynamics_for_model(
                model,
                current_state=output.current_state,
                future_action=torch.zeros_like(
                    batch["future_action"], dtype=output.current_state.dtype
                ),
                batch=batch,
            )
            image_size = (
                int(batch["current_rgb"].shape[-1]),
                int(batch["current_rgb"].shape[-2]),
            )
            cyclic_q = _decode_metric_for_model(
                model,
                predicted_state=shuffled_state,
                current_state=output.current_state,
                batch=batch,
                image_size=image_size,
            )
            zero_q = _decode_metric_for_model(
                model,
                predicted_state=zero_state,
                current_state=output.current_state,
                batch=batch,
                image_size=image_size,
            )
        difference = output.predicted_state.float() - output.target_state.float()
        shuffled_difference = shuffled_state.float() - output.target_state.float()
        copy_difference = (
            output.current_state[:, None].float() - output.target_state.float()
        )
        latent = (
            0.25 * difference[..., 0, :].square().mean(dim=-1)
            + 0.75 * difference[..., 1:, :].square().mean(dim=(-1, -2))
        ).mean(dim=-1)
        shuffle = (
            0.25 * shuffled_difference[..., 0, :].square().mean(dim=-1)
            + 0.75 * shuffled_difference[..., 1:, :].square().mean(dim=(-1, -2))
        ).mean(dim=-1)
        copy = (
            0.25 * copy_difference[..., 0, :].square().mean(dim=-1)
            + 0.75 * copy_difference[..., 1:, :].square().mean(dim=(-1, -2))
        ).mean(dim=-1)
        epe = (
            output.predicted_q_px.float() - batch["future_q_true"].float()
        ).square().sum(dim=-1).sqrt().mean(dim=(-1, -2))
        cyclic_epe = (
            cyclic_q.float() - batch["future_q_true"].float()
        ).square().sum(dim=-1).sqrt().mean(dim=(-1, -2))
        zero_epe = (
            zero_q.float() - batch["future_q_true"].float()
        ).square().sum(dim=-1).sqrt().mean(dim=(-1, -2))
        hold_q = batch["current_q_corrected"][:, None].expand_as(
            batch["future_q_true"]
        )
        hold_epe = (
            hold_q.float() - batch["future_q_true"].float()
        ).square().sum(dim=-1).sqrt().mean(dim=(-1, -2))
        predicted_aperture = torch.linalg.vector_norm(
            output.predicted_q_px[:, :, 0].float()
            - output.predicted_q_px[:, :, 1].float(),
            dim=-1,
        )
        target_aperture = torch.linalg.vector_norm(
            batch["future_q_true"][:, :, 0].float()
            - batch["future_q_true"][:, :, 1].float(),
            dim=-1,
        )
        aperture_error = (predicted_aperture - target_aperture).abs()
        target_motion = (
            batch["future_q_true"].float()
            - batch["current_q_corrected"][:, None].float()
        )
        predicted_motion = (
            output.predicted_q_px.float()
            - batch["current_q_corrected"][:, None].float()
        )
        target_norm = torch.linalg.vector_norm(target_motion, dim=-1)
        predicted_norm = torch.linalg.vector_norm(predicted_motion, dim=-1)
        direction_valid = (
            batch["future_valid"].bool()
            & (target_norm >= 1.0)
            & (predicted_norm >= 1.0e-4)
        )
        direction_cosine = (
            (target_motion * predicted_motion).sum(dim=-1)
            / (target_norm * predicted_norm).clamp_min(1.0e-6)
        ).clamp(-1.0, 1.0)
        direction_degrees = torch.rad2deg(torch.acos(direction_cosine))
        gripper_command = batch["future_action"][:, :, 3::4].mean(dim=-1)
        release_mask = gripper_command < -0.5
        for row, task_index in enumerate(batch["task_index"].cpu().tolist()):
            task = TASKS[int(task_index)]
            totals[task]["count"] += 1
            totals[task]["latent"] += float(latent[row])
            totals[task]["copy"] += float(copy[row])
            totals[task]["shuffle"] += float(shuffle[row])
            totals[task]["epe"] += float(epe[row])
            totals[task]["cyclic_epe"] += float(cyclic_epe[row])
            totals[task]["zero_epe"] += float(zero_epe[row])
            totals[task]["hold_epe"] += float(hold_epe[row])
            totals[task]["aperture"] += float(aperture_error[row].mean())
            row_direction = direction_valid[row]
            if bool(row_direction.any()):
                totals[task]["direction_degrees"] += float(
                    direction_degrees[row][row_direction].sum()
                )
                totals[task]["direction_count"] += int(row_direction.sum())
            row_release = release_mask[row]
            if bool(row_release.any()):
                totals[task]["release_aperture"] += float(
                    aperture_error[row][row_release].sum()
                )
                totals[task]["release_count"] += int(row_release.sum())
            if "camera_name" in index:
                camera = str(np.asarray(index["camera_name"])[bundles[row]])
                key = f"{task}::{camera}"
                values = view_totals.setdefault(
                    key,
                    {
                        "count": 0,
                        "epe": 0.0,
                        "cyclic_epe": 0.0,
                        "zero_epe": 0.0,
                        "hold_epe": 0.0,
                        "aperture": 0.0,
                        "direction_degrees": 0.0,
                        "direction_count": 0,
                    },
                )
                values["count"] = int(values["count"]) + 1
                values["epe"] = float(values["epe"]) + float(epe[row])
                values["cyclic_epe"] = float(values["cyclic_epe"]) + float(
                    cyclic_epe[row]
                )
                values["zero_epe"] = float(values["zero_epe"]) + float(
                    zero_epe[row]
                )
                values["hold_epe"] = float(values["hold_epe"]) + float(
                    hold_epe[row]
                )
                values["aperture"] = float(values["aperture"]) + float(
                    aperture_error[row].mean()
                )
                if bool(row_direction.any()):
                    values["direction_degrees"] = float(
                        values["direction_degrees"]
                    ) + float(direction_degrees[row][row_direction].sum())
                    values["direction_count"] = int(
                        values["direction_count"]
                    ) + int(row_direction.sum())
    per_task: dict[str, Any] = {}
    for task, values in totals.items():
        count = int(values["count"])
        if count <= 0:
            raise RuntimeError(f"joint A4 evaluation missed task {task}")
        prediction = values["latent"] / count
        per_task[task] = {
            "sequence_count": count,
            "latent_prediction_mse": prediction,
            "latent_copy_mse": values["copy"] / count,
            "action_shuffle_delta": values["shuffle"] / count - prediction,
            "anchor_epe_px": values["epe"] / count,
            "cyclic_action_anchor_epe_px": values["cyclic_epe"] / count,
            "zero_action_anchor_epe_px": values["zero_epe"] / count,
            "hold_current_anchor_epe_px": values["hold_epe"] / count,
            "gripper_aperture_mae_px": values["aperture"] / count,
            "release_aperture_mae_px": (
                values["release_aperture"] / int(values["release_count"])
                if int(values["release_count"]) > 0
                else None
            ),
            "release_horizon_count": int(values["release_count"]),
            "motion_direction_mae_degrees": (
                values["direction_degrees"] / int(values["direction_count"])
                if int(values["direction_count"]) > 0
                else None
            ),
            "motion_direction_count": int(values["direction_count"]),
        }
    per_task_view = {
        key: {
            "sequence_count": int(values["count"]),
            "anchor_epe_px": float(values["epe"]) / int(values["count"]),
            "cyclic_action_anchor_epe_px": float(values["cyclic_epe"])
            / int(values["count"]),
            "zero_action_anchor_epe_px": float(values["zero_epe"])
            / int(values["count"]),
            "hold_current_anchor_epe_px": float(values["hold_epe"])
            / int(values["count"]),
            "gripper_aperture_mae_px": float(values["aperture"])
            / int(values["count"]),
            "motion_direction_mae_degrees": (
                float(values["direction_degrees"])
                / int(values["direction_count"])
                if int(values["direction_count"]) > 0
                else None
            ),
            "motion_direction_count": int(values["direction_count"]),
        }
        for key, values in sorted(view_totals.items())
    }
    task_macro_anchor_epe = float(
        np.mean([per_task[task]["anchor_epe_px"] for task in TASKS])
    )
    task_macro_cyclic_epe = float(
        np.mean(
            [per_task[task]["cyclic_action_anchor_epe_px"] for task in TASKS]
        )
    )
    task_macro_zero_epe = float(
        np.mean([per_task[task]["zero_action_anchor_epe_px"] for task in TASKS])
    )
    task_macro_hold_epe = float(
        np.mean([per_task[task]["hold_current_anchor_epe_px"] for task in TASKS])
    )
    return {
        "sequence_count": int(len(pairs)),
        "task_macro_latent_prediction_mse": float(
            np.mean([per_task[task]["latent_prediction_mse"] for task in TASKS])
        ),
        "task_macro_action_shuffle_delta": float(
            np.mean([per_task[task]["action_shuffle_delta"] for task in TASKS])
        ),
        "task_macro_anchor_epe_px": task_macro_anchor_epe,
        "task_macro_cyclic_action_anchor_epe_px": task_macro_cyclic_epe,
        "task_macro_zero_action_anchor_epe_px": task_macro_zero_epe,
        "task_macro_hold_current_anchor_epe_px": task_macro_hold_epe,
        "task_macro_cyclic_action_epe_penalty_px": (
            task_macro_cyclic_epe - task_macro_anchor_epe
        ),
        "task_macro_zero_action_epe_penalty_px": (
            task_macro_zero_epe - task_macro_anchor_epe
        ),
        "task_macro_hold_current_epe_penalty_px": (
            task_macro_hold_epe - task_macro_anchor_epe
        ),
        "task_macro_gripper_aperture_mae_px": float(
            np.mean(
                [per_task[task]["gripper_aperture_mae_px"] for task in TASKS]
            )
        ),
        "task_macro_motion_direction_mae_degrees": float(
            np.mean(
                [
                    per_task[task]["motion_direction_mae_degrees"]
                    for task in TASKS
                    if per_task[task]["motion_direction_mae_degrees"] is not None
                ]
            )
        ),
        "per_task": per_task,
        "per_task_view": per_task_view,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser(argv)
    device = _cuda0(args.device)
    torch.cuda.set_device(device)
    seed_everything(int(args.seed))
    raw_path = args.raw_dataset.expanduser().resolve()
    index, arrays = open_raw_replay_arrays(raw_path)
    for split in ("train", "development"):
        names = set(np.asarray(index["task_name"])[np.asarray(index["split"]).astype(str) == split].astype(str))
        if names != set(TASKS):
            raise ValueError(f"joint A4 split {split!r} does not cover all seven tasks")
    statistics = input_statistics(index, arrays)
    prefix_path = args.prefix_checkpoint.expanduser().resolve()
    prefix_payload = torch.load(prefix_path, map_location="cpu", weights_only=False)
    if not isinstance(prefix_payload, Mapping):
        raise TypeError("prefix checkpoint must contain a mapping")
    model = build_joint_model_from_checkpoints(
        base_checkpoint=str(args.base_checkpoint.expanduser().resolve()),
        prefix_checkpoint=prefix_payload,
        statistics=statistics,
        seed=int(args.seed),
        loss_profile=args.loss_profile,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        _parameter_groups(model),
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1.0e-8,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup_lambda(
            step,
            min(WARMUP_STEPS, int(args.steps)),
            int(args.steps),
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("joint A4 output directory must be empty")
    resolved = {
        "protocol": JOINT_PROTOCOL,
        "seed": int(args.seed),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "raw_dataset": str(raw_path),
        "base_checkpoint": str(args.base_checkpoint.expanduser().resolve()),
        "prefix_checkpoint": str(prefix_path),
        "parameter_count": int(parameter_count),
        "sigreg_weight": SIGREG_WEIGHT,
        "metric_loss_weight": METRIC_LOSS_WEIGHT,
        "gripper_aperture_loss_weight": GRIPPER_APERTURE_LOSS_WEIGHT,
        "endpoint_metric_loss_weight": ENDPOINT_METRIC_LOSS_WEIGHT,
        "loss_profile": args.loss_profile,
        "backbone_lr": BACKBONE_LR,
        "encoder_lr": ENCODER_LR,
        "head_lr": HEAD_LR,
        "statistics": {name: value.tolist() for name, value in statistics.items()},
        "task_balanced_every_batch": True,
        "selection": "fixed_latest",
    }
    _atomic_json(output_dir / "resolved_config.json", resolved)
    print(json.dumps({"resolved": resolved}, sort_keys=True), flush=True)

    rng = np.random.default_rng(int(args.seed))
    model.train()
    started = time.perf_counter()
    final = {}
    for step in range(1, int(args.steps) + 1):
        bundles, branches = task_balanced_sample(
            index,
            split="train",
            batch_size=int(args.batch_size),
            rng=rng,
        )
        batch = _move(
            joint_batch_from_memmaps(
                index,
                arrays,
                bundle_indices=bundles,
                branch_indices=branches,
            ),
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            result = model(batch)
        if not bool(torch.isfinite(result.loss)):
            raise RuntimeError("joint A4 loss is non-finite")
        scaler.scale(result.loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        final = {
            "step": step,
            "train_loss": float(result.loss.detach()),
            "train_latent_loss": float(result.latent_loss.detach()),
            "train_metric_loss": float(result.metric_loss.detach()),
            "train_gripper_aperture_loss": float(
                result.gripper_aperture_loss.detach()
            ),
            "train_endpoint_metric_loss": float(
                result.endpoint_metric_loss.detach()
                if result.endpoint_metric_loss is not None
                else 0.0
            ),
            "train_sigreg_loss": float(result.sigreg_loss.detach()),
            "elapsed_seconds": time.perf_counter() - started,
        }
        if not all(math.isfinite(value) for value in final.values()):
            raise RuntimeError("joint A4 metrics are non-finite")
        if step % int(args.log_every) == 0 or step == int(args.steps):
            print(json.dumps(final, sort_keys=True), flush=True)
    evaluation = _evaluate(
        model,
        index,
        arrays,
        device=device,
        batch_size=int(args.eval_batch_size),
    )
    metrics = {
        **final,
        "protocol": JOINT_PROTOCOL,
        "parameter_count": int(parameter_count),
        "evaluation": evaluation,
    }
    _atomic_json(output_dir / "metrics.json", metrics)
    checkpoint = joint_checkpoint_payload(
        model,
        step=int(args.steps),
        seed=int(args.seed),
        raw_dataset=str(raw_path),
        base_checkpoint=str(args.base_checkpoint.expanduser().resolve()),
        prefix_checkpoint=str(prefix_path),
        statistics=statistics,
        loss_profile=args.loss_profile,
    )
    _atomic_torch(output_dir / "lewm_action_prefix_joint_latest.pt", checkpoint)
    print(json.dumps(metrics, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
