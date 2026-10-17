#!/usr/bin/env python3
"""Train a continuous-camera residual readout on a frozen orbit LeWM."""

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

from fcpg.lewm_action_branch_a4 import (  # noqa: E402
    HORIZON,
    SIGREG_WEIGHT,
    TASKS,
)
from fcpg.lewm_action_prefix_joint_a4 import (  # noqa: E402
    ANCHOR_COUNT,
    JOINT_PROTOCOL,
    LATENT_DIM,
    STATE_SLOTS,
    gripper_aperture_loss,
    load_joint_checkpoint,
)
from fcpg.lewm_action_prefix_joint_data import open_raw_replay_arrays  # noqa: E402
from fcpg.lewm_pose_conditioned_readout import (  # noqa: E402
    POSE_READOUT_PROTOCOL,
    REFERENCE_MODES,
    PoseConditionedResidualReadout,
    orbit_camera_basis_features,
    pose_readout_checkpoint,
)
from fcpg.lewm_runtime import cosine_warmup_lambda, seed_everything  # noqa: E402


DEFAULT_RAW = ROOT / (
    "outputs/lewm_action_prefix_a4/joint_orbit12_gripper_24train_8dev"
)
DEFAULT_BASE = ROOT / (
    "outputs/lewm_action_prefix_a4/shared_7task_orbit12_gripper_seed40531_10k/"
    "lewm_action_prefix_joint_latest.pt"
)
LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 1.0e-4
WARMUP_STEPS = 100
GRAD_CLIP = 1.0
APERTURE_LOSS_WEIGHT = 0.5
BRANCH_COUNT = 8


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--raw-dataset", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=40_541)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cache-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument(
        "--reference-mode",
        choices=REFERENCE_MODES,
        default="current_q",
        help="pixel reference corrected by the pose-conditioned readout",
    )
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


def _cuda0(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("pose readout training permits only CUDA device 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
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
        raise ValueError("pose readout requires the fixed 10k joint checkpoint")
    if float(payload.get("sigreg_weight", -1.0)) != SIGREG_WEIGHT:
        raise ValueError("base checkpoint SIGReg must be exactly 0.02")
    model = load_joint_checkpoint(payload).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, dict(payload)


def _camera_features(index: Mapping[str, np.ndarray]) -> torch.Tensor:
    required = (
        "camera_clock_hour",
        "camera_elevation_degrees",
        "camera_radius_m",
    )
    if any(name not in index for name in required):
        raise ValueError("pose readout requires continuous orbit camera metadata")
    return orbit_camera_basis_features(
        torch.from_numpy(np.asarray(index["camera_clock_hour"], dtype=np.int64)),
        torch.from_numpy(
            np.asarray(index["camera_elevation_degrees"], dtype=np.float32)
        ),
        torch.from_numpy(np.asarray(index["camera_radius_m"], dtype=np.float32)),
    )


@torch.inference_mode()
def _build_cache(
    model: nn.Module,
    index: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Encode each image once, then cache all eight frozen action rollouts."""

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
        count,
        BRANCH_COUNT,
        HORIZON,
        ANCHOR_COUNT,
        2,
        dtype=torch.float16,
    )
    task_all = torch.from_numpy(
        np.asarray(index["task_index"], dtype=np.int64)
    ).long()
    action_all = np.asarray(index["future_action"], dtype=np.float32)
    image_size = (
        int(arrays["current_rgb"].shape[-1]),
        int(arrays["current_rgb"].shape[-2]),
    )
    model.eval()
    for start in range(0, count, int(batch_size)):
        stop = min(start + int(batch_size), count)
        rows = np.arange(start, stop, dtype=np.int64)
        rgb = torch.from_numpy(
            np.ascontiguousarray(np.asarray(arrays["current_rgb"])[rows])
        )[:, None]
        proprio = torch.from_numpy(
            np.ascontiguousarray(np.asarray(arrays["current_proprio"])[rows])
        )[:, None]
        q_nominal = torch.from_numpy(
            np.ascontiguousarray(np.asarray(arrays["current_q_nominal"])[rows])
        )[:, None]
        q_corrected = torch.from_numpy(
            np.ascontiguousarray(np.asarray(arrays["current_q_corrected"])[rows])
        )[:, None]
        xyz = torch.from_numpy(
            np.ascontiguousarray(np.asarray(arrays["current_keypoints_xyz"])[rows])
        )[:, None]
        valid = torch.from_numpy(
            np.ascontiguousarray(np.asarray(arrays["current_valid"])[rows])
        )[:, None]
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            encoded = model.encode_observations(
                rgb=rgb.to(device),
                proprio=proprio.to(device),
                q_nominal=q_nominal.to(device),
                q_corrected=q_corrected.to(device),
                keypoints_xyz=xyz.to(device),
                valid=valid.to(device),
            )[:, 0]
            expanded_state = encoded[:, None].expand(
                -1, BRANCH_COUNT, -1, -1
            ).reshape(-1, STATE_SLOTS, LATENT_DIM)
            expanded_task = task_all[start:stop, None].expand(
                -1, BRANCH_COUNT
            ).reshape(-1).to(device)
            actions = torch.from_numpy(
                np.ascontiguousarray(action_all[start:stop])
            ).reshape(-1, HORIZON, 20).to(device=device, dtype=encoded.dtype)
            predicted = model.dynamics(
                expanded_state,
                actions,
                task_index=expanded_task,
            )
            grounded = q_corrected[:, 0].to(device).float()
            expanded_grounded = grounded[:, None].expand(
                -1, BRANCH_COUNT, -1, -1
            ).reshape(-1, ANCHOR_COUNT, 2)
            decoded = model.metric_readout(
                predicted,
                expanded_state,
                expanded_grounded,
                expanded_task,
                image_size=image_size,
            )
        current_state[start:stop].copy_(encoded.detach().cpu().half())
        predicted_state[start:stop].copy_(
            predicted.detach()
            .cpu()
            .half()
            .reshape(
                stop - start,
                BRANCH_COUNT,
                HORIZON,
                STATE_SLOTS,
                LATENT_DIM,
            )
        )
        base_q[start:stop].copy_(
            decoded.detach()
            .cpu()
            .half()
            .reshape(
                stop - start,
                BRANCH_COUNT,
                HORIZON,
                ANCHOR_COUNT,
                2,
            )
        )
        print(
            json.dumps(
                {"cache_rows": stop, "cache_total": count}, sort_keys=True
            ),
            flush=True,
        )
    return {
        "current_state": current_state,
        "predicted_state": predicted_state,
        "base_q": base_q,
        "current_q": torch.from_numpy(
            np.array(arrays["current_q_corrected"], dtype=np.float32, copy=True)
        ),
        "current_xyz": torch.from_numpy(
            np.array(arrays["current_keypoints_xyz"], dtype=np.float32, copy=True)
        ),
        "target_q": torch.from_numpy(
            np.array(arrays["future_q_true"], dtype=np.float32, copy=True)
        ),
        "valid": torch.from_numpy(
            np.array(arrays["future_valid"], dtype=np.bool_, copy=True)
        ),
        "task": task_all,
        "camera": _camera_features(index),
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
        "predicted_state": cache["predicted_state"][bundle, branch].to(
            device, non_blocking=True
        ),
        "current_state": cache["current_state"][bundle].to(
            device, non_blocking=True
        ),
        "base_prediction_q_px": cache["base_q"][bundle, branch].to(
            device, non_blocking=True
        ),
        "current_q_px": cache["current_q"][bundle].to(device, non_blocking=True),
        "current_keypoints_xyz": cache["current_xyz"][bundle].to(
            device, non_blocking=True
        ),
        "task_index": cache["task"][bundle].to(device, non_blocking=True),
        "camera_features": cache["camera"][bundle].to(device, non_blocking=True),
        "target_q": cache["target_q"][bundle, branch].to(
            device, non_blocking=True
        ),
        "valid": cache["valid"][bundle, branch].to(device, non_blocking=True),
        "bundle_index": bundle,
        "branch_index": branch,
    }


def _metric_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    image_size: tuple[int, int],
) -> torch.Tensor:
    width, height = image_size
    scale = prediction.new_tensor([width - 1, height - 1])
    squared = ((prediction - target) / scale).square().mean(dim=-1)
    weights = valid.float()
    return (squared * weights).sum() / weights.sum().clamp_min(1.0)


def _task_macro(values: Mapping[str, Mapping[str, float]], key: str) -> float:
    return float(np.mean([float(values[task][key]) for task in TASKS]))


def _reference_q(
    batch: Mapping[str, torch.Tensor],
    *,
    reference_mode: str,
) -> torch.Tensor:
    if reference_mode == "base_prediction":
        return batch["base_prediction_q_px"]
    if reference_mode == "current_q":
        return batch["current_q_px"][:, None].expand_as(
            batch["base_prediction_q_px"]
        )
    raise ValueError("unknown pose readout reference mode")


@torch.inference_mode()
def _evaluate(
    model: PoseConditionedResidualReadout,
    cache: Mapping[str, torch.Tensor],
    index: Mapping[str, np.ndarray],
    *,
    split: str,
    device: torch.device,
    batch_size: int,
    image_size: tuple[int, int],
    reference_mode: str,
) -> dict[str, Any]:
    model.eval()
    split_rows = np.flatnonzero(np.asarray(index["split"]).astype(str) == split)
    pair_values = (
        split_rows[:, None] * BRANCH_COUNT
        + np.arange(BRANCH_COUNT, dtype=np.int64)[None]
    ).reshape(-1)
    records: dict[str, list[np.ndarray]] = {
        "base_epe": [],
        "pose_epe": [],
        "cyclic_epe": [],
        "hold_epe": [],
        "residual": [],
        "base_angle": [],
        "pose_angle": [],
        "task": [],
        "bundle": [],
    }
    for start in range(0, pair_values.size, int(batch_size)):
        pairs_cpu = torch.from_numpy(
            np.ascontiguousarray(pair_values[start : start + int(batch_size)])
        ).long()
        batch = _pair_tensors(cache, pairs_cpu, device=device)
        output = model(
            predicted_state=batch["predicted_state"],
            current_state=batch["current_state"],
            base_prediction_q_px=_reference_q(
                batch, reference_mode=reference_mode
            ),
            current_q_px=batch["current_q_px"],
            current_keypoints_xyz=batch["current_keypoints_xyz"],
            task_index=batch["task_index"],
            camera_features=batch["camera_features"],
            image_size=image_size,
        )
        target = batch["target_q"].float()
        valid = batch["valid"].float()
        base = batch["base_prediction_q_px"].float()
        hold = batch["current_q_px"][:, None].expand_as(target)
        cyclic_branch = torch.remainder(batch["branch_index"] + 1, BRANCH_COUNT)
        cyclic_pair = batch["bundle_index"] * BRANCH_COUNT + cyclic_branch
        cyclic = _pair_tensors(cache, cyclic_pair.cpu(), device=device)
        cyclic_output = model(
            predicted_state=cyclic["predicted_state"],
            current_state=cyclic["current_state"],
            base_prediction_q_px=_reference_q(
                cyclic, reference_mode=reference_mode
            ),
            current_q_px=cyclic["current_q_px"],
            current_keypoints_xyz=cyclic["current_keypoints_xyz"],
            task_index=cyclic["task_index"],
            camera_features=cyclic["camera_features"],
            image_size=image_size,
        ).predicted_q_px

        def pair_epe(value: torch.Tensor) -> torch.Tensor:
            epe = torch.linalg.vector_norm(value.float() - target, dim=-1)
            return (epe * valid).sum(dim=(-1, -2)) / valid.sum(
                dim=(-1, -2)
            ).clamp_min(1.0)

        current_center = batch["current_q_px"].float().mean(dim=1)[:, None]
        target_delta = target.mean(dim=2) - current_center

        def angle_error(value: torch.Tensor) -> torch.Tensor:
            predicted_delta = value.float().mean(dim=2) - current_center
            target_norm = torch.linalg.vector_norm(target_delta, dim=-1)
            predicted_norm = torch.linalg.vector_norm(predicted_delta, dim=-1)
            cosine = (
                (target_delta * predicted_delta).sum(dim=-1)
                / (target_norm * predicted_norm).clamp_min(1.0e-6)
            ).clamp(-1.0, 1.0)
            angle = torch.rad2deg(torch.acos(cosine))
            return torch.where(
                (target_norm >= 4.0) & (predicted_norm >= 1.0e-3),
                angle,
                torch.full_like(angle, float("nan")),
            )

        records["base_epe"].append(pair_epe(base).cpu().numpy())
        records["pose_epe"].append(pair_epe(output.predicted_q_px).cpu().numpy())
        records["cyclic_epe"].append(pair_epe(cyclic_output).cpu().numpy())
        records["hold_epe"].append(pair_epe(hold).cpu().numpy())
        records["residual"].append(
            torch.linalg.vector_norm(output.residual_q_px, dim=-1)
            .mean(dim=(-1, -2))
            .cpu()
            .numpy()
        )
        records["base_angle"].append(angle_error(base).cpu().numpy())
        records["pose_angle"].append(
            angle_error(output.predicted_q_px).cpu().numpy()
        )
        records["task"].append(batch["task_index"].cpu().numpy())
        records["bundle"].append(batch["bundle_index"].cpu().numpy())

    merged = {name: np.concatenate(values) for name, values in records.items()}
    per_task: dict[str, Any] = {}
    per_task_view: dict[str, Any] = {}
    camera_names = np.asarray(index["camera_name"]).astype(str)
    for task_index, task in enumerate(TASKS):
        mask = merged["task"] == task_index
        per_task[task] = {
            "pair_count": int(mask.sum()),
            "base_anchor_epe_px": float(merged["base_epe"][mask].mean()),
            "pose_anchor_epe_px": float(merged["pose_epe"][mask].mean()),
            "pose_cyclic_anchor_epe_px": float(
                merged["cyclic_epe"][mask].mean()
            ),
            "hold_current_anchor_epe_px": float(
                merged["hold_epe"][mask].mean()
            ),
            "mean_pose_residual_px": float(merged["residual"][mask].mean()),
            "base_direction_mae_deg": float(
                np.nanmean(merged["base_angle"][mask])
            ),
            "pose_direction_mae_deg": float(
                np.nanmean(merged["pose_angle"][mask])
            ),
        }
        task_bundles = np.unique(merged["bundle"][mask])
        for camera in sorted(set(camera_names[task_bundles].tolist())):
            view_mask = mask & (camera_names[merged["bundle"]] == camera)
            per_task_view[f"{task}::{camera}"] = {
                "pair_count": int(view_mask.sum()),
                "base_anchor_epe_px": float(
                    merged["base_epe"][view_mask].mean()
                ),
                "pose_anchor_epe_px": float(
                    merged["pose_epe"][view_mask].mean()
                ),
                "base_direction_mae_deg": float(
                    np.nanmean(merged["base_angle"][view_mask])
                ),
                "pose_direction_mae_deg": float(
                    np.nanmean(merged["pose_angle"][view_mask])
                ),
            }
    result = {
        "split": split,
        "pair_count": int(pair_values.size),
        "task_macro_base_anchor_epe_px": _task_macro(
            per_task, "base_anchor_epe_px"
        ),
        "task_macro_pose_anchor_epe_px": _task_macro(
            per_task, "pose_anchor_epe_px"
        ),
        "task_macro_pose_cyclic_anchor_epe_px": _task_macro(
            per_task, "pose_cyclic_anchor_epe_px"
        ),
        "task_macro_hold_current_anchor_epe_px": _task_macro(
            per_task, "hold_current_anchor_epe_px"
        ),
        "task_macro_base_direction_mae_deg": _task_macro(
            per_task, "base_direction_mae_deg"
        ),
        "task_macro_pose_direction_mae_deg": _task_macro(
            per_task, "pose_direction_mae_deg"
        ),
        "per_task": per_task,
        "per_task_view": per_task_view,
    }
    result["task_macro_epe_improvement_fraction"] = 1.0 - (
        result["task_macro_pose_anchor_epe_px"]
        / result["task_macro_base_anchor_epe_px"]
    )
    result["task_macro_cyclic_penalty_px"] = (
        result["task_macro_pose_cyclic_anchor_epe_px"]
        - result["task_macro_pose_anchor_epe_px"]
    )
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
        split: sorted(
            set(
                np.asarray(index["camera_clock_hour"])[
                    np.asarray(index["split"]).astype(str) == split
                ].tolist()
            )
        )
        for split in ("train", "development")
    }
    if split_hours != {
        "train": [2, 4, 6, 8, 10, 12],
        "development": [1, 3, 5, 7, 9, 11],
    }:
        raise ValueError("orbit train/development camera split drifted")
    model, base_payload = _load_base(base_path, device=device)
    cache_started = time.perf_counter()
    cache = _build_cache(
        model,
        index,
        arrays,
        device=device,
        batch_size=int(args.cache_batch_size),
    )
    cache_seconds = time.perf_counter() - cache_started
    image_size = (
        int(arrays["current_rgb"].shape[-1]),
        int(arrays["current_rgb"].shape[-2]),
    )
    del model
    torch.cuda.empty_cache()

    readout = PoseConditionedResidualReadout(
        hidden_dim=int(args.hidden_dim)
    ).to(device)
    base_model = load_joint_checkpoint(base_payload)
    readout.initialize_embeddings_from(base_model.metric_readout)
    del base_model
    parameter_count = sum(parameter.numel() for parameter in readout.parameters())
    optimizer = torch.optim.AdamW(
        readout.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup_lambda(
            step, min(WARMUP_STEPS, int(args.steps)), int(args.steps)
        ),
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("pose readout output directory must be empty")
    resolved = {
        "protocol": POSE_READOUT_PROTOCOL,
        "seed": int(args.seed),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "hidden_dim": int(args.hidden_dim),
        "raw_dataset": str(raw_path),
        "base_checkpoint": str(base_path),
        "base_step": int(base_payload["step"]),
        "base_sigreg_weight": float(base_payload["sigreg_weight"]),
        "parameter_count": int(parameter_count),
        "learning_rate": LEARNING_RATE,
        "aperture_loss_weight": APERTURE_LOSS_WEIGHT,
        "train_camera_hours": split_hours["train"],
        "heldout_development_camera_hours": split_hours["development"],
        "selection": "fixed_latest",
        "reference_mode": str(args.reference_mode),
        "frozen_modules": ["ViT", "frame_encoder", "LeWM_dynamics", "base_readout"],
    }
    _atomic_json(output_dir / "resolved_config.json", resolved)
    print(json.dumps({"resolved": resolved}, sort_keys=True), flush=True)
    print(json.dumps({"cache_seconds": cache_seconds}, sort_keys=True), flush=True)

    split_array = np.asarray(index["split"]).astype(str)
    train_bundles = np.flatnonzero(split_array == "train")
    train_pairs = (
        train_bundles[:, None] * BRANCH_COUNT
        + np.arange(BRANCH_COUNT, dtype=np.int64)[None]
    ).reshape(-1)
    generator = torch.Generator().manual_seed(int(args.seed))
    readout.train()
    started = time.perf_counter()
    final: dict[str, Any] = {}
    for step in range(1, int(args.steps) + 1):
        selection = torch.randint(
            0,
            train_pairs.size,
            (int(args.batch_size),),
            generator=generator,
        )
        pair_indices = torch.from_numpy(train_pairs)[selection].long()
        batch = _pair_tensors(cache, pair_indices, device=device)
        optimizer.zero_grad(set_to_none=True)
        output = readout(
            predicted_state=batch["predicted_state"],
            current_state=batch["current_state"],
            base_prediction_q_px=_reference_q(
                batch, reference_mode=str(args.reference_mode)
            ),
            current_q_px=batch["current_q_px"],
            current_keypoints_xyz=batch["current_keypoints_xyz"],
            task_index=batch["task_index"],
            camera_features=batch["camera_features"],
            image_size=image_size,
        )
        metric = _metric_loss(
            output.predicted_q_px,
            batch["target_q"],
            batch["valid"],
            image_size=image_size,
        )
        aperture = gripper_aperture_loss(
            output.predicted_q_px,
            batch["target_q"],
            batch["valid"],
            image_size=image_size,
        )
        loss = metric + APERTURE_LOSS_WEIGHT * aperture
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("pose readout loss is non-finite")
        loss.backward()
        nn.utils.clip_grad_norm_(readout.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        final = {
            "step": step,
            "train_loss": float(loss.detach()),
            "train_metric_loss": float(metric.detach()),
            "train_aperture_loss": float(aperture.detach()),
            "elapsed_seconds": time.perf_counter() - started,
        }
        if not all(math.isfinite(float(value)) for value in final.values()):
            raise RuntimeError("pose readout metrics are non-finite")
        if step % int(args.log_every) == 0 or step == int(args.steps):
            print(json.dumps(final, sort_keys=True), flush=True)

    train_evaluation = _evaluate(
        readout,
        cache,
        index,
        split="train",
        device=device,
        batch_size=int(args.eval_batch_size),
        image_size=image_size,
        reference_mode=str(args.reference_mode),
    )
    development_evaluation = _evaluate(
        readout,
        cache,
        index,
        split="development",
        device=device,
        batch_size=int(args.eval_batch_size),
        image_size=image_size,
        reference_mode=str(args.reference_mode),
    )
    metrics = {
        **final,
        "protocol": POSE_READOUT_PROTOCOL,
        "parameter_count": int(parameter_count),
        "cache_seconds": float(cache_seconds),
        "train_evaluation": train_evaluation,
        "development_evaluation": development_evaluation,
    }
    _atomic_json(output_dir / "metrics.json", metrics)
    checkpoint = pose_readout_checkpoint(
        readout,
        step=int(args.steps),
        seed=int(args.seed),
        base_checkpoint=str(base_path),
        raw_dataset=str(raw_path),
        reference_mode=str(args.reference_mode),
    )
    _atomic_torch(output_dir / "lewm_pose_readout_latest.pt", checkpoint)
    print(json.dumps(metrics, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
