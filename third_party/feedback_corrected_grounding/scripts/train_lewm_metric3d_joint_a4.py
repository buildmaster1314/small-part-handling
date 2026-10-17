#!/usr/bin/env python3
"""Fine-tune the 10k camera-action LeWM with known-geometry 3D anchors."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
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

from fcpg.lewm_action_branch_a4 import SIGREG_WEIGHT, TASKS  # noqa: E402
from fcpg.lewm_action_prefix_joint_a4 import (  # noqa: E402
    joint_batch_from_memmaps,
    task_balanced_sample,
)
from fcpg.lewm_action_prefix_joint_data import open_raw_replay_arrays  # noqa: E402
from fcpg.lewm_metric3d_joint_a4 import (  # noqa: E402
    METRIC3D_PROTOCOL,
    METRIC3D_WORLD_PROTOCOL,
    PROJECTED_APERTURE_LOSS_WEIGHT,
    PROJECTED_METRIC_LOSS_WEIGHT,
    WORLD_LOSS_WEIGHT,
    build_metric3d_model,
    command_integrated_world_xyz,
    metric3d_checkpoint_payload,
)
from fcpg.lewm_runtime import cosine_warmup_lambda, seed_everything  # noqa: E402
from fcpg.lewm_world_anchor_projection import project_world_points_torch  # noqa: E402
from scripts.train_lewm_action_prefix_joint_a4 import (  # noqa: E402
    GRAD_CLIP,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    _atomic_json,
    _atomic_torch,
    _move,
)
from scripts.train_lewm_camera_action_joint_a4 import _validate_dataset  # noqa: E402


DEFAULT_RAW = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "joint_orbit12_gripper_phaseaware_a4c_v4_48train_24dev"
)
DEFAULT_PARENT = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "camera_action_phaseaware_a4c_v4_seed40571_10k/"
    "lewm_camera_action_joint_latest.pt"
)
ENCODER_LR = 1.0e-5
DYNAMICS_LR = 5.0e-5
READOUT_3D_LR = 3.0e-4


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--raw-dataset", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--parent-checkpoint", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=40_581)
    parser.add_argument("--batch-size", type=int, default=84)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--max-residual-m", type=float, default=0.15)
    parser.add_argument(
        "--residual-frame",
        choices=("camera", "world"),
        default="camera",
        help="Metric residual frame; world uses known FK/action geometry before projection.",
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--skip-final-evaluation",
        action="store_true",
        help="Only for one-step integration smoke runs.",
    )
    args = parser.parse_args(argv)
    if int(args.steps) <= 0 or int(args.seed) <= 0:
        parser.error("--steps and --seed must be positive")
    if int(args.batch_size) < len(TASKS):
        parser.error("--batch-size must be at least seven")
    if int(args.eval_batch_size) <= 0 or int(args.hidden_dim) <= 0:
        parser.error("evaluation batch size and hidden dim must be positive")
    if int(args.log_every) <= 0:
        parser.error("--log-every must be positive")
    if not 0.0 < float(args.max_residual_m) <= 0.5:
        parser.error("--max-residual-m must be in (0,0.5]")
    return args


def _cuda0(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("metric3d training permits only CUDA0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def _parameter_groups(model: nn.Module) -> list[dict[str, Any]]:
    vision = getattr(model.frame_encoder, "vision_encoder", None)
    backbone = getattr(vision, "backbone", None)
    backbone_ids: set[int] = set()
    if isinstance(backbone, nn.Module):
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
            backbone_ids.add(id(parameter))
    encoder = [
        parameter
        for module in (model.frame_encoder, model.state_projector)
        for parameter in module.parameters()
        if id(parameter) not in backbone_ids and parameter.requires_grad
    ]
    dynamics = [
        *model.dynamics.parameters(),
        *model.metric_readout.parameters(),
    ]
    readout = list(model.metric3d_readout.parameters())
    groups = [
        {"params": encoder, "lr": ENCODER_LR, "name": "grounded_encoder"},
        {"params": dynamics, "lr": DYNAMICS_LR, "name": "dynamics"},
        {"params": readout, "lr": READOUT_3D_LR, "name": "metric3d_readout"},
    ]
    flattened = [parameter for group in groups for parameter in group["params"]]
    if len(flattened) != len({id(parameter) for parameter in flattened}):
        raise RuntimeError("metric3d optimizer groups overlap")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if {id(parameter) for parameter in flattened} != {
        id(parameter) for parameter in trainable
    }:
        raise RuntimeError("metric3d optimizer does not cover trainable parameters")
    return groups


def _mean_metric(values: Mapping[str, Mapping[str, float]], key: str) -> float:
    return float(np.mean([float(values[task][key]) for task in TASKS]))


@torch.inference_mode()
def _evaluate(
    model: nn.Module,
    index: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    development = np.flatnonzero(
        np.asarray(index["split"]).astype(str) == "development"
    )
    pairs = [
        (int(bundle), branch)
        for bundle in development
        for branch in range(8)
    ]
    names = (
        "world_epe_mm",
        "static_latent_world_epe_mm",
        "shuffled_latent_world_epe_mm",
        "cyclic_world_epe_mm",
        "command_world_epe_mm",
        "pixel_epe",
        "static_latent_pixel_epe",
        "shuffled_latent_pixel_epe",
        "cyclic_pixel_epe",
        "base_2d_pixel_epe",
        "command_pixel_epe",
        "latent_mse",
    )
    totals = {
        task: {"count": 0, **{name: 0.0 for name in names}}
        for task in TASKS
    }
    per_view: dict[str, dict[str, float | int]] = {}
    for start in range(0, len(pairs), int(batch_size)):
        current = pairs[start : start + int(batch_size)]
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
                        bundles, (branches + 1) % 8
                    ]
                )
            ).to(device=device, dtype=output.current_state.dtype)
            transformed = model.transform_future_action(cyclic_action, batch)
            cyclic_state = model.dynamics(
                output.current_state,
                transformed.to(dtype=output.current_state.dtype),
                task_index=batch["task_index"],
            )
            cyclic = model.decode_3d(
                predicted_state=cyclic_state,
                current_state=output.current_state,
                current_keypoints_xyz=batch["current_keypoints_xyz"],
                future_action=cyclic_action,
                task_index=batch["task_index"],
                camera_intrinsic=batch["camera_intrinsic"],
                extrinsic_cam_from_world=batch[
                    "camera_extrinsic_cam_from_world"
                ],
            )
            static_state = output.current_state[:, None].expand_as(
                output.predicted_state
            )
            static_latent = model.decode_3d(
                predicted_state=static_state,
                current_state=output.current_state,
                current_keypoints_xyz=batch["current_keypoints_xyz"],
                future_action=batch["future_action"],
                task_index=batch["task_index"],
                camera_intrinsic=batch["camera_intrinsic"],
                extrinsic_cam_from_world=batch[
                    "camera_extrinsic_cam_from_world"
                ],
            )
            shuffled_latent = model.decode_3d(
                predicted_state=torch.roll(output.predicted_state, shifts=1, dims=0),
                current_state=output.current_state,
                current_keypoints_xyz=batch["current_keypoints_xyz"],
                future_action=batch["future_action"],
                task_index=batch["task_index"],
                camera_intrinsic=batch["camera_intrinsic"],
                extrinsic_cam_from_world=batch[
                    "camera_extrinsic_cam_from_world"
                ],
            )
            command_world = command_integrated_world_xyz(
                batch["current_keypoints_xyz"], batch["future_action"]
            )
            command_q = project_world_points_torch(
                command_world,
                batch["camera_intrinsic"],
                batch["camera_extrinsic_cam_from_world"],
            ).pixels

        target_world = batch["future_keypoints_xyz"].float()
        target_q = batch["future_q_true"].float()
        valid = batch["future_valid"].bool()

        def world_epe(value: torch.Tensor) -> torch.Tensor:
            error = 1000.0 * torch.linalg.vector_norm(
                value.float() - target_world, dim=-1
            )
            weights = valid.float()
            return (error * weights).sum(dim=(-1, -2)) / weights.sum(
                dim=(-1, -2)
            ).clamp_min(1.0)

        def pixel_epe(value: torch.Tensor) -> torch.Tensor:
            error = torch.linalg.vector_norm(value.float() - target_q, dim=-1)
            weights = valid.float()
            return (error * weights).sum(dim=(-1, -2)) / weights.sum(
                dim=(-1, -2)
            ).clamp_min(1.0)

        latent_difference = output.predicted_state.float() - output.target_state.float()
        latent_mse = (
            0.25 * latent_difference[:, :, 0].square().mean(dim=-1)
            + 0.75 * latent_difference[:, :, 1:].square().mean(dim=(-1, -2))
        ).mean(dim=-1)
        values = {
            "world_epe_mm": world_epe(output.predicted_world_xyz),
            "static_latent_world_epe_mm": world_epe(
                static_latent.predicted_world_xyz
            ),
            "shuffled_latent_world_epe_mm": world_epe(
                shuffled_latent.predicted_world_xyz
            ),
            "cyclic_world_epe_mm": world_epe(cyclic.predicted_world_xyz),
            "command_world_epe_mm": world_epe(command_world),
            "pixel_epe": pixel_epe(output.predicted_q_px),
            "static_latent_pixel_epe": pixel_epe(
                static_latent.predicted_q_px
            ),
            "shuffled_latent_pixel_epe": pixel_epe(
                shuffled_latent.predicted_q_px
            ),
            "cyclic_pixel_epe": pixel_epe(cyclic.predicted_q_px),
            "base_2d_pixel_epe": pixel_epe(output.base_predicted_q_px),
            "command_pixel_epe": pixel_epe(command_q),
            "latent_mse": latent_mse,
        }
        for row, task_number in enumerate(batch["task_index"].cpu().tolist()):
            task = TASKS[int(task_number)]
            totals[task]["count"] = int(totals[task]["count"]) + 1
            for name, metric in values.items():
                totals[task][name] = float(totals[task][name]) + float(metric[row])
            camera = str(np.asarray(index["camera_name"])[bundles[row]])
            key = f"{task}::{camera}"
            view = per_view.setdefault(
                key,
                {
                    "count": 0,
                    "world_epe_mm": 0.0,
                    "pixel_epe": 0.0,
                    "base_2d_pixel_epe": 0.0,
                },
            )
            view["count"] = int(view["count"]) + 1
            for name in ("world_epe_mm", "pixel_epe", "base_2d_pixel_epe"):
                view[name] = float(view[name]) + float(values[name][row])

    per_task: dict[str, dict[str, float | int]] = {}
    for task in TASKS:
        count = int(totals[task]["count"])
        if count <= 0:
            raise RuntimeError(f"metric3d evaluation missed task {task}")
        per_task[task] = {"sequence_count": count}
        for name in names:
            per_task[task][name] = float(totals[task][name]) / count
    normalized_views = {
        key: {
            "sequence_count": int(value["count"]),
            "world_epe_mm": float(value["world_epe_mm"]) / int(value["count"]),
            "pixel_epe": float(value["pixel_epe"]) / int(value["count"]),
            "base_2d_pixel_epe": float(value["base_2d_pixel_epe"])
            / int(value["count"]),
        }
        for key, value in sorted(per_view.items())
    }
    result: dict[str, Any] = {
        "sequence_count": len(pairs),
        "per_task": per_task,
        "per_task_view": normalized_views,
    }
    for name in names:
        result[f"task_macro_{name}"] = _mean_metric(per_task, name)
    result["task_macro_cyclic_world_penalty_mm"] = (
        result["task_macro_cyclic_world_epe_mm"]
        - result["task_macro_world_epe_mm"]
    )
    result["task_macro_cyclic_pixel_penalty_px"] = (
        result["task_macro_cyclic_pixel_epe"] - result["task_macro_pixel_epe"]
    )
    result["task_macro_static_latent_world_penalty_mm"] = (
        result["task_macro_static_latent_world_epe_mm"]
        - result["task_macro_world_epe_mm"]
    )
    result["task_macro_shuffled_latent_world_penalty_mm"] = (
        result["task_macro_shuffled_latent_world_epe_mm"]
        - result["task_macro_world_epe_mm"]
    )
    result["task_macro_static_latent_pixel_penalty_px"] = (
        result["task_macro_static_latent_pixel_epe"]
        - result["task_macro_pixel_epe"]
    )
    result["task_macro_shuffled_latent_pixel_penalty_px"] = (
        result["task_macro_shuffled_latent_pixel_epe"]
        - result["task_macro_pixel_epe"]
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser(argv)
    device = _cuda0(args.device)
    torch.cuda.set_device(device)
    seed_everything(int(args.seed))
    torch.backends.cuda.matmul.allow_tf32 = True
    raw_path = args.raw_dataset.expanduser().resolve()
    parent_path = args.parent_checkpoint.expanduser().resolve()
    index, arrays = open_raw_replay_arrays(raw_path)
    _validate_dataset(index, arrays)
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if not isinstance(parent, Mapping):
        raise TypeError("metric3d parent checkpoint must contain a mapping")
    checkpoint_raw = Path(str(parent.get("raw_dataset", ""))).resolve()
    if checkpoint_raw != raw_path:
        raise ValueError("metric3d parent and training dataset differ")
    model = build_metric3d_model(
        parent,
        seed=int(args.seed),
        hidden_dim=int(args.hidden_dim),
        max_residual_m=float(args.max_residual_m),
        residual_frame=str(args.residual_frame),
    ).to(device)
    groups = _parameter_groups(model)
    optimizer = torch.optim.AdamW(
        groups,
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
        raise FileExistsError("metric3d output directory must be empty")
    resolved = {
        "protocol": (
            METRIC3D_WORLD_PROTOCOL
            if args.residual_frame == "world"
            else METRIC3D_PROTOCOL
        ),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "raw_dataset": str(raw_path),
        "parent_checkpoint": str(parent_path),
        "parent_step": int(parent["step"]),
        "sigreg_weight": SIGREG_WEIGHT,
        "hidden_dim": int(args.hidden_dim),
        "max_residual_m": float(args.max_residual_m),
        "residual_frame": str(args.residual_frame),
        "world_loss_weight": WORLD_LOSS_WEIGHT,
        "projected_metric_loss_weight": PROJECTED_METRIC_LOSS_WEIGHT,
        "projected_aperture_loss_weight": PROJECTED_APERTURE_LOSS_WEIGHT,
        "encoder_lr": ENCODER_LR,
        "dynamics_lr": DYNAMICS_LR,
        "readout_3d_lr": READOUT_3D_LR,
        "vision_backbone_frozen": True,
        "coordinate_contract": (
            "world_frame_known_geometry_reference_plus_bounded_residual_then_exact_projection"
            if args.residual_frame == "world"
            else "camera_frame_3d_residual_then_exact_world_projection"
        ),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameter_count": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
    }
    _atomic_json(output_dir / "resolved_config.json", resolved)
    print(json.dumps({"resolved": resolved}, sort_keys=True), flush=True)

    rng = np.random.default_rng(int(args.seed))
    model.train()
    started = time.perf_counter()
    final: dict[str, Any] = {}
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
            output = model(batch)
        if not bool(torch.isfinite(output.loss)):
            raise RuntimeError("metric3d loss is non-finite")
        scaler.scale(output.loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        final = {
            "step": step,
            "train_loss": float(output.loss.detach()),
            "train_base_loss": float(output.base_loss.detach()),
            "train_latent_loss": float(output.latent_loss.detach()),
            "train_world_loss": float(output.world_loss.detach()),
            "train_projected_metric_loss": float(output.metric_loss.detach()),
            "train_projected_aperture_loss": float(
                output.gripper_aperture_loss.detach()
            ),
            "train_sigreg_loss": float(output.sigreg_loss.detach()),
            "elapsed_seconds": time.perf_counter() - started,
        }
        if not all(math.isfinite(float(value)) for value in final.values()):
            raise RuntimeError("metric3d training metric is non-finite")
        if step % int(args.log_every) == 0 or step == int(args.steps):
            print(json.dumps(final, sort_keys=True), flush=True)

    evaluation = (
        None
        if bool(args.skip_final_evaluation)
        else _evaluate(
            model,
            index,
            arrays,
            device=device,
            batch_size=int(args.eval_batch_size),
        )
    )
    metrics = {
        **final,
        "protocol": (
            METRIC3D_WORLD_PROTOCOL
            if args.residual_frame == "world"
            else METRIC3D_PROTOCOL
        ),
        "evaluation": evaluation,
    }
    _atomic_json(output_dir / "metrics.json", metrics)
    _atomic_torch(
        output_dir / "lewm_metric3d_joint_latest.pt",
        metric3d_checkpoint_payload(
            model,
            step=int(args.steps),
            seed=int(args.seed),
            parent_checkpoint=str(parent_path),
            raw_dataset=str(raw_path),
            protocol=(
                METRIC3D_WORLD_PROTOCOL
                if args.residual_frame == "world"
                else METRIC3D_PROTOCOL
            ),
        ),
    )
    print(json.dumps(metrics, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
