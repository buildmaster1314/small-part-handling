#!/usr/bin/env python3
"""Train a frozen-LeWM world-anchor head and project it into held-out views."""

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
    ANCHOR_COUNT,
    JOINT_PROTOCOL,
    LATENT_DIM,
    STATE_SLOTS,
    load_joint_checkpoint,
)
from fcpg.lewm_action_prefix_joint_data import open_raw_replay_arrays  # noqa: E402
from fcpg.lewm_runtime import cosine_warmup_lambda, seed_everything  # noqa: E402
from fcpg.lewm_world_anchor_projection import (  # noqa: E402
    WORLD_ANCHOR_PROTOCOL,
    WorldAnchorDynamicsReadout,
    project_world_points_torch,
    world_anchor_checkpoint,
)


DEFAULT_RAW = ROOT / "outputs/lewm_action_prefix_a4/joint_orbit12_gripper_24train_8dev"
DEFAULT_BASE = ROOT / (
    "outputs/lewm_action_prefix_a4/shared_7task_orbit12_gripper_seed40531_10k/"
    "lewm_action_prefix_joint_latest.pt"
)
BRANCH_COUNT = 8
LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 1.0e-4
WARMUP_STEPS = 100
GRAD_CLIP = 1.0
WORLD_SCALE_M = 0.1
PIXEL_LOSS_WEIGHT = 1.0
CROSS_VIEW_CONSISTENCY_WEIGHT = 0.5


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--raw-dataset", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=40_561)
    parser.add_argument("--batch-size", type=int, default=504)
    parser.add_argument("--cache-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    for name in (
        "steps",
        "seed",
        "batch_size",
        "cache_batch_size",
        "eval_batch_size",
        "hidden_dim",
        "log_every",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def multiview_training_groups(
    index: Mapping[str, np.ndarray],
    *,
    split: str = "train",
) -> np.ndarray:
    """Return physical bundles as equal-size groups of camera rows."""

    splits = np.asarray(index["split"]).astype(str)
    physical = np.asarray(index["physical_bundle_index"], dtype=np.int64)
    selected = np.flatnonzero(splits == str(split))
    groups = [
        np.flatnonzero((splits == str(split)) & (physical == physical_id))
        for physical_id in np.unique(physical[selected])
    ]
    if not groups or any(group.size != 6 for group in groups):
        raise ValueError("world-anchor training requires six views per physical bundle")
    return np.stack(groups).astype(np.int64, copy=False)


def _cuda0(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("world-anchor training permits only CUDA device 0")
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


def _load_base(path: Path, *, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("protocol") != JOINT_PROTOCOL:
        raise ValueError("base checkpoint is not a joint A4 LeWM")
    if int(payload.get("step", -1)) != 10_000:
        raise ValueError("world-anchor readout requires the fixed 10k joint checkpoint")
    if float(payload.get("sigreg_weight", -1.0)) != SIGREG_WEIGHT:
        raise ValueError("base checkpoint SIGReg must be exactly 0.02")
    model = load_joint_checkpoint(payload).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, dict(payload)


@torch.inference_mode()
def _build_cache(
    model: nn.Module,
    index: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    required = ("camera_intrinsic", "camera_extrinsic_cam_from_world")
    if any(name not in arrays for name in required):
        raise ValueError(
            "world-anchor training requires cached camera matrices; run "
            "cache_lewm_orbit_camera_matrices.py first"
        )
    count = int(np.asarray(index["task_index"]).shape[0])
    current_state = torch.empty(count, STATE_SLOTS, LATENT_DIM, dtype=torch.float16)
    predicted_state = torch.empty(
        count,
        BRANCH_COUNT,
        HORIZON,
        STATE_SLOTS,
        LATENT_DIM,
        dtype=torch.float16,
    )
    base_q = torch.empty(
        count, BRANCH_COUNT, HORIZON, ANCHOR_COUNT, 2, dtype=torch.float16
    )
    task_all = torch.from_numpy(np.asarray(index["task_index"], dtype=np.int64)).long()
    action_all = np.asarray(index["future_action"], dtype=np.float32)
    image_size = (int(arrays["current_rgb"].shape[-1]), int(arrays["current_rgb"].shape[-2]))
    for start in range(0, count, int(batch_size)):
        stop = min(start + int(batch_size), count)
        rows = np.arange(start, stop, dtype=np.int64)
        values = {
            name: torch.from_numpy(np.ascontiguousarray(np.asarray(arrays[name])[rows]))[:, None]
            for name in (
                "current_rgb",
                "current_proprio",
                "current_q_nominal",
                "current_q_corrected",
                "current_keypoints_xyz",
                "current_valid",
            )
        }
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            encoded = model.encode_observations(
                rgb=values["current_rgb"].to(device),
                proprio=values["current_proprio"].to(device),
                q_nominal=values["current_q_nominal"].to(device),
                q_corrected=values["current_q_corrected"].to(device),
                keypoints_xyz=values["current_keypoints_xyz"].to(device),
                valid=values["current_valid"].to(device),
            )[:, 0]
            expanded_state = encoded[:, None].expand(-1, BRANCH_COUNT, -1, -1).reshape(
                -1, STATE_SLOTS, LATENT_DIM
            )
            expanded_task = task_all[start:stop, None].expand(-1, BRANCH_COUNT).reshape(-1).to(device)
            actions = torch.from_numpy(np.ascontiguousarray(action_all[start:stop])).reshape(
                -1, HORIZON, 20
            ).to(device=device, dtype=encoded.dtype)
            predicted = model.dynamics(expanded_state, actions, task_index=expanded_task)
            current_q = values["current_q_corrected"][:, 0].to(device).float()
            expanded_q = current_q[:, None].expand(-1, BRANCH_COUNT, -1, -1).reshape(
                -1, ANCHOR_COUNT, 2
            )
            decoded = model.metric_readout(
                predicted,
                expanded_state,
                expanded_q,
                expanded_task,
                image_size=image_size,
            )
        current_state[start:stop].copy_(encoded.detach().cpu().half())
        predicted_state[start:stop].copy_(
            predicted.detach().cpu().half().reshape(
                stop - start, BRANCH_COUNT, HORIZON, STATE_SLOTS, LATENT_DIM
            )
        )
        base_q[start:stop].copy_(
            decoded.detach().cpu().half().reshape(
                stop - start, BRANCH_COUNT, HORIZON, ANCHOR_COUNT, 2
            )
        )
        print(json.dumps({"cache_rows": stop, "cache_total": count}), flush=True)
    return {
        "current_state": current_state,
        "predicted_state": predicted_state,
        "base_q": base_q,
        "current_xyz": torch.from_numpy(np.array(arrays["current_keypoints_xyz"], dtype=np.float32, copy=True)),
        "target_xyz": torch.from_numpy(np.array(arrays["future_keypoints_xyz"], dtype=np.float32, copy=True)),
        "target_q": torch.from_numpy(np.array(arrays["future_q_true"], dtype=np.float32, copy=True)),
        "valid": torch.from_numpy(np.array(arrays["future_valid"], dtype=np.bool_, copy=True)),
        "action": torch.from_numpy(np.array(index["future_action"], dtype=np.float32, copy=True)),
        "task": task_all,
        "intrinsic": torch.from_numpy(np.array(arrays["camera_intrinsic"], dtype=np.float32, copy=True)),
        "extrinsic": torch.from_numpy(np.array(arrays["camera_extrinsic_cam_from_world"], dtype=np.float32, copy=True)),
    }


def _pair_tensors(
    cache: Mapping[str, torch.Tensor],
    pair_indices: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    bundle = torch.div(pair_indices, BRANCH_COUNT, rounding_mode="floor")
    branch = torch.remainder(pair_indices, BRANCH_COUNT)
    return {
        "predicted_state": cache["predicted_state"][bundle, branch].to(device, non_blocking=True),
        "current_state": cache["current_state"][bundle].to(device, non_blocking=True),
        "base_q": cache["base_q"][bundle, branch].to(device, non_blocking=True),
        "current_xyz": cache["current_xyz"][bundle].to(device, non_blocking=True),
        "target_xyz": cache["target_xyz"][bundle, branch].to(device, non_blocking=True),
        "target_q": cache["target_q"][bundle, branch].to(device, non_blocking=True),
        "valid": cache["valid"][bundle, branch].to(device, non_blocking=True),
        "action": cache["action"][bundle, branch].to(device, non_blocking=True),
        "task": cache["task"][bundle].to(device, non_blocking=True),
        "intrinsic": cache["intrinsic"][bundle].to(device, non_blocking=True),
        "extrinsic": cache["extrinsic"][bundle].to(device, non_blocking=True),
        "bundle": bundle,
        "branch": branch,
    }


def command_integrated_xyz(current_xyz: torch.Tensor, future_action: torch.Tensor) -> torch.Tensor:
    """Translate all anchors by causal MetaWorld Cartesian target commands."""

    if current_xyz.ndim != 3 or current_xyz.shape[1:] != (ANCHOR_COUNT, 3):
        raise ValueError("current_xyz must have shape [B,3,3]")
    if future_action.shape != (current_xyz.shape[0], HORIZON, 20):
        raise ValueError("future_action must have shape [B,8,20]")
    raw = future_action.float().reshape(-1, HORIZON, 5, 4)
    horizon_delta = 0.01 * raw[..., :3].sum(dim=2)
    cumulative = torch.cumsum(horizon_delta, dim=1)
    return current_xyz[:, None].float() + cumulative[:, :, None, :]


def _losses(
    prediction_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    prediction_q: torch.Tensor,
    target_q: torch.Tensor,
    valid: torch.Tensor,
    *,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    world = ((prediction_xyz - target_xyz) / WORLD_SCALE_M).square().mean()
    scale = prediction_q.new_tensor([image_size[0] - 1, image_size[1] - 1])
    squared = ((prediction_q - target_q) / scale).square().mean(dim=-1)
    weights = valid.float()
    pixel = (squared * weights).sum() / weights.sum().clamp_min(1.0)
    return world + PIXEL_LOSS_WEIGHT * pixel, world, pixel


def _task_macro(per_task: Mapping[str, Mapping[str, float]], key: str) -> float:
    return float(np.mean([float(per_task[task][key]) for task in TASKS]))


@torch.inference_mode()
def _evaluate(
    model: WorldAnchorDynamicsReadout,
    cache: Mapping[str, torch.Tensor],
    index: Mapping[str, np.ndarray],
    *,
    split: str,
    device: torch.device,
    batch_size: int,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    model.eval()
    split_rows = np.flatnonzero(np.asarray(index["split"]).astype(str) == split)
    pair_values = (
        split_rows[:, None] * BRANCH_COUNT + np.arange(BRANCH_COUNT, dtype=np.int64)[None]
    ).reshape(-1)
    records: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "world_epe_mm",
            "cyclic_world_epe_mm",
            "hold_world_epe_mm",
            "command_world_epe_mm",
            "pixel_epe",
            "cyclic_pixel_epe",
            "base_pixel_epe",
            "hold_pixel_epe",
            "command_pixel_epe",
            "prediction_xyz",
            "task",
            "bundle",
            "branch",
        )
    }
    for start in range(0, pair_values.size, int(batch_size)):
        pairs = torch.from_numpy(np.ascontiguousarray(pair_values[start : start + int(batch_size)])).long()
        batch = _pair_tensors(cache, pairs, device=device)
        command_xyz = command_integrated_xyz(batch["current_xyz"], batch["action"])
        output = model(
            predicted_state=batch["predicted_state"],
            current_state=batch["current_state"],
            current_keypoints_xyz=batch["current_xyz"],
            camera_rotation_world_to_camera=batch["extrinsic"][:, :3, :3],
            task_index=batch["task"],
            reference_keypoints_xyz=command_xyz,
        )
        projected = project_world_points_torch(
            output.predicted_xyz_world, batch["intrinsic"], batch["extrinsic"]
        ).pixels
        cyclic_branch = torch.remainder(batch["branch"] + 1, BRANCH_COUNT)
        cyclic_pairs = batch["bundle"] * BRANCH_COUNT + cyclic_branch
        cyclic_batch = _pair_tensors(cache, cyclic_pairs.cpu(), device=device)
        cyclic_output = model(
            predicted_state=cyclic_batch["predicted_state"],
            current_state=batch["current_state"],
            current_keypoints_xyz=batch["current_xyz"],
            camera_rotation_world_to_camera=batch["extrinsic"][:, :3, :3],
            task_index=batch["task"],
            reference_keypoints_xyz=command_xyz,
        )
        cyclic_q = project_world_points_torch(
            cyclic_output.predicted_xyz_world, batch["intrinsic"], batch["extrinsic"]
        ).pixels
        hold_xyz = batch["current_xyz"][:, None].expand_as(batch["target_xyz"])
        hold_q = project_world_points_torch(
            hold_xyz, batch["intrinsic"], batch["extrinsic"]
        ).pixels
        command_q = project_world_points_torch(
            command_xyz, batch["intrinsic"], batch["extrinsic"]
        ).pixels

        def world_epe(value: torch.Tensor) -> torch.Tensor:
            return 1000.0 * torch.linalg.vector_norm(
                value.float() - batch["target_xyz"].float(), dim=-1
            ).mean(dim=(-1, -2))

        def pixel_epe(value: torch.Tensor) -> torch.Tensor:
            error = torch.linalg.vector_norm(
                value.float() - batch["target_q"].float(), dim=-1
            )
            valid = batch["valid"].float()
            return (error * valid).sum(dim=(-1, -2)) / valid.sum(dim=(-1, -2)).clamp_min(1.0)

        for name, value in (
            ("world_epe_mm", world_epe(output.predicted_xyz_world)),
            ("cyclic_world_epe_mm", world_epe(cyclic_output.predicted_xyz_world)),
            ("hold_world_epe_mm", world_epe(hold_xyz)),
            ("command_world_epe_mm", world_epe(command_xyz)),
            ("pixel_epe", pixel_epe(projected)),
            ("cyclic_pixel_epe", pixel_epe(cyclic_q)),
            ("base_pixel_epe", pixel_epe(batch["base_q"])),
            ("hold_pixel_epe", pixel_epe(hold_q)),
            ("command_pixel_epe", pixel_epe(command_q)),
        ):
            records[name].append(value.cpu().numpy())
        records["prediction_xyz"].append(output.predicted_xyz_world.cpu().numpy())
        records["task"].append(batch["task"].cpu().numpy())
        records["bundle"].append(batch["bundle"].cpu().numpy())
        records["branch"].append(batch["branch"].cpu().numpy())

    merged = {name: np.concatenate(values) for name, values in records.items()}
    per_task: dict[str, dict[str, float | int]] = {}
    metric_names = (
        "world_epe_mm",
        "cyclic_world_epe_mm",
        "hold_world_epe_mm",
        "command_world_epe_mm",
        "pixel_epe",
        "cyclic_pixel_epe",
        "base_pixel_epe",
        "hold_pixel_epe",
        "command_pixel_epe",
    )
    for task_index, task in enumerate(TASKS):
        mask = merged["task"] == task_index
        per_task[task] = {"pair_count": int(mask.sum())}
        for name in metric_names:
            per_task[task][name] = float(merged[name][mask].mean())
    view_std_values = []
    physical = np.asarray(index["physical_bundle_index"], dtype=np.int64)
    for physical_id in np.unique(physical[split_rows]):
        rows = np.flatnonzero((physical == physical_id) & (np.asarray(index["split"]).astype(str) == split))
        for branch in range(BRANCH_COUNT):
            mask = np.isin(merged["bundle"], rows) & (merged["branch"] == branch)
            if int(mask.sum()) >= 2:
                view_std_values.append(
                    float(1000.0 * merged["prediction_xyz"][mask].std(axis=0).mean())
                )
    result: dict[str, Any] = {
        "split": split,
        "pair_count": int(pair_values.size),
        "per_task": per_task,
        "mean_cross_view_prediction_std_mm": float(np.mean(view_std_values)),
    }
    for name in metric_names:
        result[f"task_macro_{name}"] = _task_macro(per_task, name)
    result["task_macro_cyclic_world_penalty_mm"] = (
        result["task_macro_cyclic_world_epe_mm"] - result["task_macro_world_epe_mm"]
    )
    result["task_macro_cyclic_pixel_penalty_px"] = (
        result["task_macro_cyclic_pixel_epe"] - result["task_macro_pixel_epe"]
    )
    result["gate"] = {
        "beats_command_integration_3d": bool(
            result["task_macro_world_epe_mm"] < result["task_macro_command_world_epe_mm"]
        ),
        "beats_base_2d_readout": bool(
            result["task_macro_pixel_epe"] < result["task_macro_base_pixel_epe"]
        ),
        "uses_action_conditioned_state": bool(
            result["task_macro_cyclic_world_penalty_mm"] > 0.0
        ),
    }
    result["gate"]["go"] = bool(all(result["gate"].values()))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser(argv)
    device = _cuda0(args.device)
    torch.cuda.set_device(device)
    seed_everything(int(args.seed))
    torch.backends.cuda.matmul.allow_tf32 = True
    raw_path = args.raw_dataset.expanduser().resolve()
    base_path = args.base_checkpoint.expanduser().resolve()
    index, arrays = open_raw_replay_arrays(raw_path)
    split_hours = {
        split: sorted(set(np.asarray(index["camera_clock_hour"])[np.asarray(index["split"]).astype(str) == split].tolist()))
        for split in ("train", "development")
    }
    if split_hours != {"train": [2, 4, 6, 8, 10, 12], "development": [1, 3, 5, 7, 9, 11]}:
        raise ValueError("orbit train/development camera split drifted")
    base_model, base_payload = _load_base(base_path, device=device)
    cache_started = time.perf_counter()
    cache = _build_cache(
        base_model,
        index,
        arrays,
        device=device,
        batch_size=int(args.cache_batch_size),
    )
    cache_seconds = time.perf_counter() - cache_started
    model = WorldAnchorDynamicsReadout(hidden_dim=int(args.hidden_dim)).to(device)
    model.initialize_embeddings_from(base_model.metric_readout)
    del base_model
    torch.cuda.empty_cache()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup_lambda(step, min(WARMUP_STEPS, int(args.steps)), int(args.steps)),
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("world-anchor output directory must be empty")
    resolved = {
        "protocol": WORLD_ANCHOR_PROTOCOL,
        "seed": int(args.seed),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "hidden_dim": int(args.hidden_dim),
        "raw_dataset": str(raw_path),
        "base_checkpoint": str(base_path),
        "base_step": int(base_payload["step"]),
        "base_sigreg_weight": float(base_payload["sigreg_weight"]),
        "parameter_count": int(parameter_count),
        "train_camera_hours": split_hours["train"],
        "heldout_development_camera_hours": split_hours["development"],
        "world_scale_m": WORLD_SCALE_M,
        "pixel_loss_weight": PIXEL_LOSS_WEIGHT,
        "cross_view_consistency_weight": CROSS_VIEW_CONSISTENCY_WEIGHT,
        "camera_conditioning": "continuous_world_to_camera_rotation_3x3",
        "selection": "fixed_latest",
        "frozen_modules": ["ViT", "frame_encoder", "LeWM_dynamics", "base_2d_readout"],
    }
    _atomic_json(output_dir / "resolved_config.json", resolved)
    print(json.dumps({"resolved": resolved, "cache_seconds": cache_seconds}, sort_keys=True), flush=True)

    train_groups = multiview_training_groups(index)
    view_count = int(train_groups.shape[1])
    if int(args.batch_size) % view_count != 0:
        raise ValueError("batch-size must be divisible by six camera views")
    groups_per_batch = int(args.batch_size) // view_count
    generator = torch.Generator().manual_seed(int(args.seed))
    image_size = (int(arrays["current_rgb"].shape[-1]), int(arrays["current_rgb"].shape[-2]))
    started = time.perf_counter()
    final: dict[str, Any] = {}
    model.train()
    for step in range(1, int(args.steps) + 1):
        group_selection = torch.randint(
            0, train_groups.shape[0], (groups_per_batch,), generator=generator
        ).numpy()
        branch_selection = torch.randint(
            0, BRANCH_COUNT, (groups_per_batch,), generator=generator
        ).numpy()
        selected_rows = train_groups[group_selection]
        pairs = torch.from_numpy(
            np.ascontiguousarray(
                selected_rows * BRANCH_COUNT + branch_selection[:, None]
            ).reshape(-1)
        ).long()
        batch = _pair_tensors(cache, pairs, device=device)
        command_xyz = command_integrated_xyz(batch["current_xyz"], batch["action"])
        optimizer.zero_grad(set_to_none=True)
        output = model(
            predicted_state=batch["predicted_state"],
            current_state=batch["current_state"],
            current_keypoints_xyz=batch["current_xyz"],
            camera_rotation_world_to_camera=batch["extrinsic"][:, :3, :3],
            task_index=batch["task"],
            reference_keypoints_xyz=command_xyz,
        )
        projected = project_world_points_torch(
            output.predicted_xyz_world, batch["intrinsic"], batch["extrinsic"]
        ).pixels
        prediction_grouped = output.predicted_xyz_world.reshape(
            groups_per_batch, view_count, HORIZON, ANCHOR_COUNT, 3
        )
        consistency_loss = (
            (
                prediction_grouped
                - prediction_grouped.mean(dim=1, keepdim=True)
            )
            / WORLD_SCALE_M
        ).square().mean()
        supervised_loss, world_loss, pixel_loss = _losses(
            output.predicted_xyz_world,
            batch["target_xyz"].float(),
            projected,
            batch["target_q"].float(),
            batch["valid"],
            image_size=image_size,
        )
        loss = supervised_loss + CROSS_VIEW_CONSISTENCY_WEIGHT * consistency_loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("world-anchor loss is non-finite")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        final = {
            "step": step,
            "train_loss": float(loss.detach()),
            "train_world_loss": float(world_loss.detach()),
            "train_pixel_loss": float(pixel_loss.detach()),
            "train_cross_view_consistency_loss": float(
                consistency_loss.detach()
            ),
            "elapsed_seconds": time.perf_counter() - started,
        }
        if not all(math.isfinite(float(value)) for value in final.values()):
            raise RuntimeError("world-anchor metrics are non-finite")
        if step % int(args.log_every) == 0 or step == int(args.steps):
            print(json.dumps(final, sort_keys=True), flush=True)

    train_eval = _evaluate(
        model, cache, index, split="train", device=device,
        batch_size=int(args.eval_batch_size), image_size=image_size,
    )
    development_eval = _evaluate(
        model, cache, index, split="development", device=device,
        batch_size=int(args.eval_batch_size), image_size=image_size,
    )
    metrics = {
        **final,
        "protocol": WORLD_ANCHOR_PROTOCOL,
        "parameter_count": int(parameter_count),
        "cache_seconds": float(cache_seconds),
        "train_evaluation": train_eval,
        "development_evaluation": development_eval,
    }
    _atomic_json(output_dir / "metrics.json", metrics)
    _atomic_torch(
        output_dir / "lewm_world_anchor_latest.pt",
        world_anchor_checkpoint(
            model,
            step=int(args.steps),
            seed=int(args.seed),
            base_checkpoint=str(base_path),
            raw_dataset=str(raw_path),
        ),
    )
    print(json.dumps(metrics, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
