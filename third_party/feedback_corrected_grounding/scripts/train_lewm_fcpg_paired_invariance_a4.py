#!/usr/bin/env python3
"""Continue a seven-task LeWM with FCPG-supervised paired-view invariance."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_prefix_joint_a4 import (  # noqa: E402
    SIGREG_WEIGHT,
    input_statistics,
    joint_batch_from_memmaps,
    joint_checkpoint_payload,
    load_joint_checkpoint,
    task_balanced_sample,
)
from fcpg.lewm_action_prefix_joint_data import open_raw_replay_arrays  # noqa: E402
from fcpg.lewm_runtime import cosine_warmup_lambda  # noqa: E402
from fcpg.lewm_fcpg_paired_invariance import (  # noqa: E402
    GROUNDING_MODES,
    PROTOCOL,
    canonical_task_indices,
    clean_identity_loss,
    grounded_valid_mask,
    paired_alignment_loss,
    repeated_action_prefix,
    select_dynamic_grounding,
)
from scripts.train_lewm_action_prefix_joint_a4 import (  # noqa: E402
    _evaluate,
    _move,
)


DEFAULT_SOURCE = ROOT / (
    "outputs/lewm_action_prefix_a4/shared_7task_orbit12_gripper_seed40531_10k/"
    "lewm_action_prefix_joint_latest.pt"
)
DEFAULT_TRAIN_DATA = ROOT / (
    "outputs/lewm_action_branch_a4/paired_view_canonicalizer_v1/data/"
    "train_static_seed61011.npz"
)
DEFAULT_TRAIN_CACHE = ROOT / (
    "outputs/lewm_action_branch_a4/paired_view_canonicalizer_v1/cache/"
    "train_static_cache.npz"
)
DEFAULT_DEV_LINEAR_DATA = ROOT / (
    "outputs/lewm_action_branch_a4/paired_view_canonicalizer_v1/data/"
    "dev_linear_seed62012.npz"
)
DEFAULT_DEV_LINEAR_CACHE = ROOT / (
    "outputs/lewm_action_branch_a4/paired_view_canonicalizer_v1/cache/"
    "dev_linear_cache.npz"
)
DEFAULT_DEV_STEP_DATA = ROOT / (
    "outputs/lewm_action_branch_a4/paired_view_canonicalizer_v1/data/"
    "dev_step_seed63013.npz"
)
DEFAULT_DEV_STEP_CACHE = ROOT / (
    "outputs/lewm_action_branch_a4/paired_view_canonicalizer_v1/cache/"
    "dev_step_cache.npz"
)


@dataclass(frozen=True)
class PairedFrames:
    dynamic_rgb: np.ndarray
    clean_rgb: np.ndarray
    proprio: np.ndarray
    q_nominal: np.ndarray
    q_fcpg: np.ndarray
    q_true: np.ndarray
    keypoints_xyz: np.ndarray
    valid_nominal: np.ndarray
    valid_true: np.ndarray
    raw_action: np.ndarray
    task_index: np.ndarray
    task_name: np.ndarray
    episode_id: np.ndarray
    frame_id: np.ndarray
    camera_step_frame: np.ndarray
    image_size: tuple[int, int]

    @property
    def count(self) -> int:
        return int(self.dynamic_rgb.shape[0])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--raw-dataset", type=Path, default=None)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--train-cache", type=Path, default=DEFAULT_TRAIN_CACHE)
    parser.add_argument(
        "--eval-pair",
        action="append",
        default=[],
        help="Named evaluation pair as name=data.npz,cache.npz.",
    )
    parser.add_argument("--grounding", choices=GROUNDING_MODES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=40621)
    parser.add_argument("--base-batch-size", type=int, default=42)
    parser.add_argument("--pair-batch-size", type=int, default=24)
    parser.add_argument("--eval-batch-size", type=int, default=84)
    parser.add_argument("--pair-eval-batch-size", type=int, default=64)
    parser.add_argument("--backbone-lr", type=float, default=2e-6)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pair-weight", type=float, default=0.05)
    parser.add_argument("--clean-identity-weight", type=float, default=0.5)
    parser.add_argument("--summary-weight", type=float, default=0.25)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _default_eval_pairs() -> dict[str, tuple[Path, Path]]:
    return {
        "linear": (DEFAULT_DEV_LINEAR_DATA, DEFAULT_DEV_LINEAR_CACHE),
        "step": (DEFAULT_DEV_STEP_DATA, DEFAULT_DEV_STEP_CACHE),
    }


def _parse_eval_pairs(values: list[str]) -> dict[str, tuple[Path, Path]]:
    if not values:
        return _default_eval_pairs()
    result: dict[str, tuple[Path, Path]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--eval-pair must use name=data.npz,cache.npz")
        name, paths = value.split("=", 1)
        pieces = paths.split(",")
        if not name or name in result or len(pieces) != 2:
            raise ValueError("evaluation pair names and paths must be unique and complete")
        result[name] = (Path(pieces[0]), Path(pieces[1]))
    return result


def _cuda0(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("paired-invariance training permits only CUDA device 0")
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


def _load_paired_frames(data_path: Path, cache_path: Path) -> PairedFrames:
    with np.load(data_path, allow_pickle=False) as data, np.load(
        cache_path, allow_pickle=False
    ) as cache:
        required_data = {
            "rgb",
            "paired_clean_rgb",
            "obs",
            "q_nominal",
            "q_true",
            "keypoints_xyz",
            "valid_nominal",
            "valid_true",
            "action",
            "task_name",
            "episode_id",
            "frame_id",
            "camera_step_frame",
            "image_width",
            "image_height",
        }
        required_cache = {
            "q_dynamic_corrected",
            "task_name",
            "episode_id",
            "frame_id",
        }
        if not required_data.issubset(data.files):
            raise ValueError("paired-view data is missing required arrays")
        if not required_cache.issubset(cache.files):
            raise ValueError("paired-view cache is missing required arrays")
        for name in ("task_name", "episode_id", "frame_id"):
            if not np.array_equal(data[name], cache[name]):
                raise ValueError(f"paired-view data/cache identity mismatch for {name}")
        dynamic_rgb = np.array(data["rgb"], copy=True)
        clean_rgb = np.array(data["paired_clean_rgb"], copy=True)
        q_nominal = np.asarray(data["q_nominal"], dtype=np.float32)
        q_fcpg = np.asarray(cache["q_dynamic_corrected"], dtype=np.float32)
        if dynamic_rgb.shape != clean_rgb.shape or q_nominal.shape != q_fcpg.shape:
            raise ValueError("paired dynamic/clean arrays do not align")
        task_name = np.asarray(data["task_name"]).astype(str)
        return PairedFrames(
            dynamic_rgb=dynamic_rgb,
            clean_rgb=clean_rgb,
            proprio=np.asarray(data["obs"], dtype=np.float32)[:, :4].copy(),
            q_nominal=q_nominal.copy(),
            q_fcpg=q_fcpg.copy(),
            q_true=np.asarray(data["q_true"], dtype=np.float32).copy(),
            keypoints_xyz=np.asarray(data["keypoints_xyz"], dtype=np.float32).copy(),
            valid_nominal=np.asarray(data["valid_nominal"], dtype=bool).copy(),
            valid_true=np.asarray(data["valid_true"], dtype=bool).copy(),
            raw_action=np.asarray(data["action"], dtype=np.float32).copy(),
            task_index=canonical_task_indices(task_name),
            task_name=task_name.copy(),
            episode_id=np.asarray(data["episode_id"], dtype=np.int64).copy(),
            frame_id=np.asarray(data["frame_id"], dtype=np.int64).copy(),
            camera_step_frame=np.asarray(
                data["camera_step_frame"], dtype=np.int64
            ).copy(),
            image_size=(int(data["image_width"].item()), int(data["image_height"].item())),
        )


def _frame_tensors(
    frames: PairedFrames,
    rows: np.ndarray,
    *,
    image_key: str,
    grounding: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if image_key not in {"dynamic", "clean"}:
        raise ValueError("image_key must be dynamic or clean")
    indices = np.asarray(rows, dtype=np.int64)
    if image_key == "dynamic":
        rgb = frames.dynamic_rgb[indices]
        if grounding == "oracle":
            grounded = frames.q_true[indices]
            base_valid = frames.valid_true[indices]
        else:
            grounded = select_dynamic_grounding(
                frames.q_nominal[indices], frames.q_fcpg[indices], mode=grounding
            )
            base_valid = frames.valid_nominal[indices]
    else:
        rgb = frames.clean_rgb[indices]
        grounded = frames.q_nominal[indices]
        base_valid = frames.valid_nominal[indices]
    valid = grounded_valid_mask(
        base_valid, grounded, image_size=frames.image_size
    )
    return {
        "rgb": torch.from_numpy(
            np.ascontiguousarray(np.transpose(rgb, (0, 3, 1, 2)))
        ).to(device, non_blocking=True),
        "proprio": torch.from_numpy(
            np.ascontiguousarray(frames.proprio[indices])
        ).to(device, non_blocking=True),
        "q_nominal": torch.from_numpy(
            np.ascontiguousarray(frames.q_nominal[indices])
        ).to(device, non_blocking=True),
        "q_grounded": torch.from_numpy(np.ascontiguousarray(grounded)).to(
            device, non_blocking=True
        ),
        "xyz": torch.from_numpy(
            np.ascontiguousarray(frames.keypoints_xyz[indices])
        ).to(device, non_blocking=True),
        "valid": torch.from_numpy(np.ascontiguousarray(valid)).to(
            device, non_blocking=True
        ),
    }


def _encode_frames(
    model: nn.Module,
    tensors: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    return model.encode_observations(
        rgb=tensors["rgb"][:, None],
        proprio=tensors["proprio"][:, None],
        q_nominal=tensors["q_nominal"][:, None],
        q_corrected=tensors["q_grounded"][:, None],
        keypoints_xyz=tensors["xyz"][:, None],
        valid=tensors["valid"][:, None],
    )[:, 0]


@torch.inference_mode()
def _teacher_clean_states(
    model: nn.Module,
    frames: PairedFrames,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    for start in range(0, frames.count, batch_size):
        rows = np.arange(start, min(start + batch_size, frames.count))
        tensors = _frame_tensors(
            frames, rows, image_key="clean", grounding="nominal", device=device
        )
        parts.append(_encode_frames(model, tensors).float().cpu().numpy())
    return np.concatenate(parts).astype(np.float32, copy=False)


@torch.inference_mode()
def _evaluate_pairs(
    model: nn.Module,
    frames: PairedFrames,
    teacher_clean: np.ndarray,
    *,
    grounding: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    totals = {
        task: {
            "count": 0,
            "state": 0.0,
            "anchor": 0.0,
            "summary": 0.0,
            "forecast": 0.0,
            "forecast_anchor": 0.0,
            "clean": 0.0,
        }
        for task in sorted(set(frames.task_name.tolist()))
    }
    for start in range(0, frames.count, batch_size):
        rows = np.arange(start, min(start + batch_size, frames.count))
        dynamic = _encode_frames(
            model,
            _frame_tensors(
                frames,
                rows,
                image_key="dynamic",
                grounding=grounding,
                device=device,
            ),
        )
        clean = _encode_frames(
            model,
            _frame_tensors(
                frames,
                rows,
                image_key="clean",
                grounding="nominal",
                device=device,
            ),
        )
        teacher = torch.from_numpy(np.ascontiguousarray(teacher_clean[rows])).to(
            device
        )
        raw_action = torch.from_numpy(
            np.ascontiguousarray(frames.raw_action[rows])
        ).to(device)
        task_index = torch.from_numpy(
            np.ascontiguousarray(frames.task_index[rows])
        ).to(device)
        prefix = repeated_action_prefix(raw_action).to(dtype=dynamic.dtype)
        dynamic_forecast = model.dynamics(dynamic, prefix, task_index=task_index)
        teacher_forecast = model.dynamics(teacher, prefix, task_index=task_index)
        state_distance = torch.linalg.vector_norm(
            dynamic.float() - teacher.float(), dim=-1
        )
        forecast_distance = torch.linalg.vector_norm(
            dynamic_forecast.float() - teacher_forecast.float(), dim=-1
        )
        clean_distance = torch.linalg.vector_norm(
            clean.float() - teacher.float(), dim=-1
        )
        batch_metrics = torch.stack(
            (
                state_distance.mean(dim=1),
                state_distance[:, 1:].mean(dim=1),
                state_distance[:, :1].mean(dim=1),
                forecast_distance.mean(dim=(1, 2)),
                forecast_distance[:, :, 1:].mean(dim=(1, 2)),
                clean_distance.mean(dim=1),
            ),
            dim=1,
        ).cpu().numpy()
        batch_tasks = frames.task_name[rows].astype(str)
        for task in np.unique(batch_tasks):
            selected = batch_metrics[batch_tasks == task]
            values = totals[str(task)]
            values["count"] += int(selected.shape[0])
            values["state"] += float(selected[:, 0].sum())
            values["anchor"] += float(selected[:, 1].sum())
            values["summary"] += float(selected[:, 2].sum())
            values["forecast"] += float(selected[:, 3].sum())
            values["forecast_anchor"] += float(selected[:, 4].sum())
            values["clean"] += float(selected[:, 5].sum())
    per_task = {
        task: {
            "frame_count": int(values["count"]),
            "state_gap": float(values["state"]) / int(values["count"]),
            "anchor_gap": float(values["anchor"]) / int(values["count"]),
            "summary_gap": float(values["summary"]) / int(values["count"]),
            "forecast_gap": float(values["forecast"]) / int(values["count"]),
            "forecast_anchor_gap": float(values["forecast_anchor"])
            / int(values["count"]),
            "clean_identity_gap": float(values["clean"]) / int(values["count"]),
        }
        for task, values in totals.items()
    }
    keys = (
        "state_gap",
        "anchor_gap",
        "summary_gap",
        "forecast_gap",
        "forecast_anchor_gap",
        "clean_identity_gap",
    )
    return {
        "frame_count": frames.count,
        "grounding": grounding,
        "task_macro": {
            key: float(np.mean([per_task[task][key] for task in per_task]))
            for key in keys
        },
        "per_task": per_task,
    }


def _encoder_parameter_groups(
    model: nn.Module,
    *,
    backbone_lr: float,
    encoder_lr: float,
) -> list[dict[str, Any]]:
    model.requires_grad_(False)
    model.frame_encoder.requires_grad_(True)
    model.state_projector.requires_grad_(True)
    vision = getattr(model.frame_encoder, "vision_encoder", None)
    backbone = getattr(vision, "backbone", None)
    backbone_parameters = (
        list(backbone.parameters()) if isinstance(backbone, nn.Module) else []
    )
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    encoder_parameters = [
        parameter
        for module in (model.frame_encoder, model.state_projector)
        for parameter in module.parameters()
        if id(parameter) not in backbone_ids
    ]
    groups: list[dict[str, Any]] = []
    if backbone_parameters:
        groups.append(
            {"params": backbone_parameters, "lr": float(backbone_lr), "name": "backbone"}
        )
    groups.append(
        {"params": encoder_parameters, "lr": float(encoder_lr), "name": "encoder"}
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    flattened = [parameter for group in groups for parameter in group["params"]]
    if {id(value) for value in flattened} != {id(value) for value in trainable}:
        raise RuntimeError("paired-invariance optimizer does not cover trainable parameters")
    return groups


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.steps <= 0 or args.base_batch_size <= 0 or args.pair_batch_size <= 0:
        raise ValueError("training steps and batch sizes must be positive")
    if args.base_batch_size < 7:
        raise ValueError("base batch size must cover all seven tasks")
    if float(args.pair_weight) <= 0.0 or float(args.clean_identity_weight) < 0.0:
        raise ValueError("loss weights are invalid")
    if float(args.backbone_lr) <= 0.0 or float(args.encoder_lr) <= 0.0:
        raise ValueError("learning rates must be positive")
    if int(args.warmup_steps) < 0:
        raise ValueError("warmup steps must be non-negative")
    device = _cuda0(str(args.device))
    torch.cuda.set_device(device)
    _seed_everything(int(args.seed))

    source_path = args.source_checkpoint.expanduser().resolve()
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(source, Mapping):
        raise TypeError("source checkpoint must contain a mapping")
    if int(source.get("step", -1)) != 10_000:
        raise ValueError("paired-invariance source must be the fixed 10k endpoint")
    if float(source.get("sigreg_weight", -1.0)) != SIGREG_WEIGHT:
        raise ValueError("paired-invariance SIGReg must remain exactly 0.02")
    raw_path = (
        args.raw_dataset.expanduser().resolve()
        if args.raw_dataset is not None
        else Path(str(source.get("raw_dataset", ""))).expanduser().resolve()
    )
    if raw_path != Path(str(source.get("raw_dataset", ""))).expanduser().resolve():
        raise ValueError("raw dataset must match the source checkpoint")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("output directory must be empty")

    train_frames = _load_paired_frames(
        args.train_data.expanduser().resolve(), args.train_cache.expanduser().resolve()
    )
    eval_paths = _parse_eval_pairs(list(args.eval_pair))
    eval_frames = {
        name: _load_paired_frames(data.resolve(), cache.resolve())
        for name, (data, cache) in eval_paths.items()
    }
    model = load_joint_checkpoint(source).to(device)
    raw_index, raw_arrays = open_raw_replay_arrays(raw_path)
    expected_statistics = input_statistics(raw_index, raw_arrays)
    for name, expected in expected_statistics.items():
        actual = np.asarray(source["statistics"][name], dtype=np.float32)
        if not np.allclose(actual, expected, rtol=0.0, atol=1e-6):
            raise ValueError(f"source input statistics drifted for {name}")

    teacher_train = _teacher_clean_states(
        model,
        train_frames,
        device=device,
        batch_size=int(args.pair_eval_batch_size),
    )
    teacher_eval = {
        name: _teacher_clean_states(
            model,
            frames,
            device=device,
            batch_size=int(args.pair_eval_batch_size),
        )
        for name, frames in eval_frames.items()
    }
    source_pair_evaluation = {
        name: {
            mode: _evaluate_pairs(
                model,
                frames,
                teacher_eval[name],
                grounding=mode,
                device=device,
                batch_size=int(args.pair_eval_batch_size),
            )
            for mode in GROUNDING_MODES
        }
        for name, frames in eval_frames.items()
    }

    optimizer = torch.optim.AdamW(
        _encoder_parameter_groups(
            model,
            backbone_lr=float(args.backbone_lr),
            encoder_lr=float(args.encoder_lr),
        ),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup_lambda(
            step,
            int(args.warmup_steps),
            int(args.steps),
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)
    rng = np.random.default_rng(int(args.seed))

    calibration_rows = np.arange(min(256, train_frames.count), dtype=np.int64)
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        calibration_dynamic = _encode_frames(
            model,
            _frame_tensors(
                train_frames,
                calibration_rows,
                image_key="dynamic",
                grounding=str(args.grounding),
                device=device,
            ),
        )
        calibration_teacher = torch.from_numpy(
            np.ascontiguousarray(teacher_train[calibration_rows])
        ).to(device)
        pair_scale = float(
            paired_alignment_loss(
                calibration_dynamic.float(),
                calibration_teacher.float(),
                summary_weight=float(args.summary_weight),
            )
            .clamp_min(1e-6)
            .cpu()
        )

    resolved = {
        "protocol": PROTOCOL,
        "source_checkpoint": str(source_path),
        "source_step": int(source["step"]),
        "grounding": str(args.grounding),
        "steps": int(args.steps),
        "seed": int(args.seed),
        "sigreg_weight": SIGREG_WEIGHT,
        "raw_dataset": str(raw_path),
        "train_data": str(args.train_data.expanduser().resolve()),
        "train_cache": str(args.train_cache.expanduser().resolve()),
        "eval_pairs": {
            name: {"data": str(data.resolve()), "cache": str(cache.resolve())}
            for name, (data, cache) in eval_paths.items()
        },
        "base_batch_size": int(args.base_batch_size),
        "pair_batch_size": int(args.pair_batch_size),
        "backbone_lr": float(args.backbone_lr),
        "encoder_lr": float(args.encoder_lr),
        "warmup_steps": int(args.warmup_steps),
        "pair_weight": float(args.pair_weight),
        "clean_identity_weight": float(args.clean_identity_weight),
        "summary_weight": float(args.summary_weight),
        "pair_scale": pair_scale,
        "trainable_scope": "frame_encoder_and_state_projector_only",
        "training_mode": "deterministic_eval_with_gradients",
        "teacher": "source_10k_clean_same_physical_state_stop_gradient",
        "future_information_used": False,
        "camera_extrinsic_input_used": False,
        "cuda_device": 0,
    }
    _atomic_json(output_dir / "resolved_config.json", resolved)
    print(json.dumps({"resolved": resolved}, sort_keys=True), flush=True)

    # Gradients remain enabled in eval mode. Keeping dropout disabled makes the
    # frozen source checkpoint an exact clean-state teacher during adaptation.
    model.eval()
    started = time.perf_counter()
    logs: list[dict[str, float | int]] = []
    for step in range(1, int(args.steps) + 1):
        bundles, branches = task_balanced_sample(
            raw_index,
            split="train",
            batch_size=int(args.base_batch_size),
            rng=rng,
        )
        base_batch = _move(
            joint_batch_from_memmaps(
                raw_index,
                raw_arrays,
                bundle_indices=bundles,
                branch_indices=branches,
            ),
            device,
        )
        pair_rows = rng.integers(
            0, train_frames.count, size=int(args.pair_batch_size), dtype=np.int64
        )
        dynamic_tensors = _frame_tensors(
            train_frames,
            pair_rows,
            image_key="dynamic",
            grounding=str(args.grounding),
            device=device,
        )
        clean_tensors = _frame_tensors(
            train_frames,
            pair_rows,
            image_key="clean",
            grounding="nominal",
            device=device,
        )
        teacher = torch.from_numpy(
            np.ascontiguousarray(teacher_train[pair_rows])
        ).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            base_output = model(base_batch)
            dynamic_state = _encode_frames(model, dynamic_tensors)
            clean_state = _encode_frames(model, clean_tensors)
            pair_loss = paired_alignment_loss(
                dynamic_state.float(),
                teacher.float(),
                summary_weight=float(args.summary_weight),
            )
            identity_loss = clean_identity_loss(clean_state.float(), teacher.float())
            normalized_pair = pair_loss / pair_scale
            normalized_identity = identity_loss / pair_scale
            loss = (
                base_output.loss
                + float(args.pair_weight) * normalized_pair
                + float(args.clean_identity_weight) * normalized_identity
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("paired-invariance loss is non-finite")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            1.0,
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError("paired-invariance gradient is non-finite")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step == 1 or step % int(args.log_every) == 0 or step == int(args.steps):
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "base_loss": float(base_output.loss.detach()),
                "pair_loss": float(pair_loss.detach()),
                "clean_identity_loss": float(identity_loss.detach()),
                "normalized_pair_loss": float(normalized_pair.detach()),
                "normalized_clean_identity_loss": float(normalized_identity.detach()),
                "gradient_norm": float(gradient_norm.detach()),
                "elapsed_seconds": time.perf_counter() - started,
            }
            if not all(math.isfinite(float(value)) for value in row.values()):
                raise RuntimeError("paired-invariance log contains non-finite values")
            logs.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    final_pair_evaluation = {
        name: {
            mode: _evaluate_pairs(
                model,
                frames,
                teacher_eval[name],
                grounding=mode,
                device=device,
                batch_size=int(args.pair_eval_batch_size),
            )
            for mode in GROUNDING_MODES
        }
        for name, frames in eval_frames.items()
    }
    final_a4_evaluation = _evaluate(
        model,
        raw_index,
        raw_arrays,
        device=device,
        batch_size=int(args.eval_batch_size),
    )
    source_metrics_path = source_path.parent / "metrics.json"
    source_a4_evaluation = None
    if source_metrics_path.exists():
        source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
        source_a4_evaluation = source_metrics.get("evaluation")

    metrics = {
        "protocol": PROTOCOL,
        "grounding": str(args.grounding),
        "source_step": int(source["step"]),
        "final_step": int(source["step"]) + int(args.steps),
        "sigreg_weight": SIGREG_WEIGHT,
        "logs": logs,
        "source_pair_evaluation": source_pair_evaluation,
        "final_pair_evaluation": final_pair_evaluation,
        "source_a4_evaluation": source_a4_evaluation,
        "final_a4_evaluation": final_a4_evaluation,
    }
    _atomic_json(output_dir / "metrics.json", metrics)

    statistics = {
        name: np.asarray(source["statistics"][name], dtype=np.float32)
        for name in ("proprio_mean", "proprio_std", "action_mean", "action_std")
    }
    payload = joint_checkpoint_payload(
        model,
        step=int(source["step"]) + int(args.steps),
        seed=int(args.seed),
        raw_dataset=str(raw_path),
        base_checkpoint=str(source["base_checkpoint"]),
        prefix_checkpoint=str(source["prefix_checkpoint"]),
        statistics=statistics,
        loss_profile=str(source.get("loss_profile", "uniform_v1")),
    )
    payload["paired_invariance_protocol"] = PROTOCOL
    payload["paired_invariance_grounding"] = str(args.grounding)
    payload["paired_invariance_source"] = str(source_path)
    payload["paired_invariance_additional_steps"] = int(args.steps)
    _atomic_torch(output_dir / "lewm_action_prefix_joint_latest.pt", payload)
    print(json.dumps(metrics, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
